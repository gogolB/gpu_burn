#include "spatial_redundancy.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <random>

// Device function to get current SM ID
__device__ int getSMID() {
    uint ret;
    asm("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

// Device function to force execution on specific SM (simplified)
__device__ void forceSMExecution(int targetSM) {
    // This is a simplified approach - real implementation would require
    // more sophisticated SM scheduling control
    while (getSMID() != targetSM) {
        __threadfence();
    }
}

// Spatial execution kernel
__global__ void spatialExecutionKernel(void* output, const void* input, 
                                      size_t startIdx, size_t endIdx,
                                      int targetSM) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = startIdx + tid;
    
    if (idx >= endIdx) return;
    
    // Simple example: copy with validation
    // In practice, this would call the actual kernel computation
    float* out = static_cast<float*>(output);
    const float* in = static_cast<const float*>(input);
    
    // Store SM ID for validation
    if (tid == 0) {
        int currentSM = getSMID();
        // Could store SM info for validation
    }
    
    out[idx] = in[idx];
}

// Merge partitions kernel
__global__ void mergePartitionsKernel(void* output, void** partitions,
                                     size_t* partitionSizes, int numPartitions,
                                     size_t totalElements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalElements) return;
    
    // Find which partition this element belongs to
    size_t cumSize = 0;
    int partitionIdx = 0;
    for (int i = 0; i < numPartitions; i++) {
        if (tid < cumSize + partitionSizes[i]) {
            partitionIdx = i;
            break;
        }
        cumSize += partitionSizes[i];
    }
    
    // Copy from partition to merged output
    size_t localIdx = tid - cumSize;
    float* out = static_cast<float*>(output);
    float* partitionData = static_cast<float*>(partitions[partitionIdx]);
    out[tid] = partitionData[localIdx];
}

// Compare partitions kernel
__global__ void comparePartitionsKernel(void** partitions, size_t numElements,
                                       int numPartitions, bool* hasErrors) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numElements) return;
    
    float* partition0 = static_cast<float*>(partitions[0]);
    float val0 = partition0[tid];
    
    for (int i = 1; i < numPartitions; i++) {
        float* partition = static_cast<float*>(partitions[i]);
        if (partition[tid] != val0) {
            *hasErrors = true;
            return;
        }
    }
}

// SMProfiler implementation
SMProfiler::SMProfiler() {}

SMProfiler::~SMProfiler() {
    for (auto& pair : profileData_) {
        cudaEventDestroy(pair.second.startEvent);
        cudaEventDestroy(pair.second.endEvent);
    }
}

void SMProfiler::startProfiling(int smId) {
    if (profileData_.find(smId) == profileData_.end()) {
        ProfileData data;
        cudaEventCreate(&data.startEvent);
        cudaEventCreate(&data.endEvent);
        profileData_[smId] = data;
    }
    cudaEventRecord(profileData_[smId].startEvent);
}

void SMProfiler::endProfiling(int smId) {
    if (profileData_.find(smId) != profileData_.end()) {
        cudaEventRecord(profileData_[smId].endEvent);
        cudaEventSynchronize(profileData_[smId].endEvent);
        
        float ms;
        cudaEventElapsedTime(&ms, profileData_[smId].startEvent, 
                           profileData_[smId].endEvent);
        profileData_[smId].executionTimes.push_back(ms);
    }
}

double SMProfiler::getAverageTime(int smId) const {
    auto it = profileData_.find(smId);
    if (it != profileData_.end() && !it->second.executionTimes.empty()) {
        double sum = 0.0;
        for (float time : it->second.executionTimes) {
            sum += time;
        }
        return sum / it->second.executionTimes.size();
    }
    return 0.0;
}

size_t SMProfiler::getExecutionCount(int smId) const {
    auto it = profileData_.find(smId);
    if (it != profileData_.end()) {
        return it->second.executionTimes.size();
    }
    return 0;
}

// SpatialRedundancyValidator implementation
SpatialRedundancyValidator::SpatialRedundancyValidator(
    SpatialStrategy strategy, SMSelectionMode smMode)
    : strategy_(strategy)
    , smMode_(smMode)
    , smMask_(0xFFFFFFFFFFFFFFFF)
    , d_mergedOutput_(nullptr)
    , bufferSize_(0) {
    
    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProps_, device);
    numSMs_ = deviceProps_.multiProcessorCount;
    maxThreadsPerSM_ = deviceProps_.maxThreadsPerMultiProcessor;
    
    // Initialize SM stats
    smStats_.resize(numSMs_);
    for (int i = 0; i < numSMs_; i++) {
        smStats_[i].smId = i;
    }
}

SpatialRedundancyValidator::~SpatialRedundancyValidator() {
    cleanup();
}

void SpatialRedundancyValidator::setup(const KernelConfig& config) {
    // Select SMs based on mode
    selectSMs();
    
    // Calculate buffer size
    size_t requiredSize = config.matrixSize * config.matrixSize * sizeof(float);
    
    if (bufferSize_ < requiredSize) {
        cleanup();
        
        // Allocate partition outputs based on strategy
        if (strategy_ == SpatialStrategy::PARTITION_SM) {
            d_partitionOutputs_.resize(selectedSMs_.size());
            for (size_t i = 0; i < selectedSMs_.size(); i++) {
                cudaMalloc(&d_partitionOutputs_[i], requiredSize / selectedSMs_.size() + 1024);
            }
        } else {
            // For replicated strategies, each SM gets full output buffer
            d_partitionOutputs_.resize(selectedSMs_.size());
            for (size_t i = 0; i < selectedSMs_.size(); i++) {
                cudaMalloc(&d_partitionOutputs_[i], requiredSize);
            }
        }
        
        cudaMalloc(&d_mergedOutput_, requiredSize);
        bufferSize_ = requiredSize;
    }
}

void SpatialRedundancyValidator::cleanup() {
    for (void* ptr : d_partitionOutputs_) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    d_partitionOutputs_.clear();
    
    if (d_mergedOutput_) {
        cudaFree(d_mergedOutput_);
        d_mergedOutput_ = nullptr;
    }
    
    bufferSize_ = 0;
}

ValidationResult SpatialRedundancyValidator::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& config) {
    
    ValidationResult result;
    result.method = ValidationType::SPATIAL_REDUNDANCY;
    
    if (!kernelFunc_) {
        result.passed = false;
        result.errorDetails = "No kernel function set for spatial redundancy validation";
        return result;
    }
    
    // Execute based on strategy
    switch (strategy_) {
        case SpatialStrategy::PARTITION_SM:
            executePartitionedSM(data, numElements, config);
            mergePartitionedResults(d_mergedOutput_, numElements, elementSize);
            result.passed = true;
            break;
            
        case SpatialStrategy::REPLICATE_SM:
            executeReplicatedSM(data, numElements, config);
            result.passed = compareReplicatedResults(numElements, elementSize);
            if (!result.passed) {
                result.errorDetails = "Spatial redundancy detected inconsistent results across SMs";
            }
            break;
            
        case SpatialStrategy::CHECKERBOARD:
            executeCheckerboard(data, numElements, config);
            result.passed = true;
            break;
            
        case SpatialStrategy::ROUND_ROBIN:
            executeRoundRobin(data, numElements, config);
            result.passed = true;
            break;
    }
    
    // Copy result back
    if (result.passed) {
        cudaMemcpy(const_cast<void*>(data), d_mergedOutput_, 
                  numElements * elementSize, cudaMemcpyDeviceToDevice);
    }
    
    // Update SM statistics
    for (auto& stat : smStats_) {
        if (stat.hasErrors) {
            stat.errorCount++;
            result.corruptedElements++;
        }
    }
    
    if (result.corruptedElements > 0) {
        result.confidence = 1.0 - (static_cast<double>(result.corruptedElements) / selectedSMs_.size());
        result.errorDetails += " (" + std::to_string(result.corruptedElements) + 
                             " SMs reported errors)";
    }
    
    return result;
}

void SpatialRedundancyValidator::selectSMs() {
    selectedSMs_.clear();
    
    switch (smMode_) {
        case SMSelectionMode::ALL_SMS:
            for (int i = 0; i < numSMs_; i++) {
                selectedSMs_.push_back(i);
            }
            break;
            
        case SMSelectionMode::ALTERNATE_SMS:
            for (int i = 0; i < numSMs_; i += 2) {
                selectedSMs_.push_back(i);
            }
            break;
            
        case SMSelectionMode::RANDOM_SMS:
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> dis(0, numSMs_ - 1);
                
                int numSelected = numSMs_ / 2;  // Select half
                while (selectedSMs_.size() < numSelected) {
                    int sm = dis(gen);
                    if (std::find(selectedSMs_.begin(), selectedSMs_.end(), sm) == selectedSMs_.end()) {
                        selectedSMs_.push_back(sm);
                    }
                }
            }
            break;
            
        case SMSelectionMode::MANUAL_SMS:
            for (int i = 0; i < numSMs_; i++) {
                if (smMask_ & (1ULL << i)) {
                    selectedSMs_.push_back(i);
                }
            }
            break;
    }
}

void SpatialRedundancyValidator::executePartitionedSM(
    const void* input, size_t numElements, const KernelConfig& config) {
    
    std::vector<size_t> startIndices, endIndices;
    partitionWork(numElements, startIndices, endIndices);
    
    // Execute kernel on each selected SM with its partition
    for (size_t i = 0; i < selectedSMs_.size(); i++) {
        int smId = selectedSMs_[i];
        size_t partitionSize = endIndices[i] - startIndices[i];
        
        // Calculate grid/block for this partition
        int blockSize = 256;
        int gridSize = (partitionSize + blockSize - 1) / blockSize;
        
        // Execute spatial kernel
        spatialExecutionKernel<<<gridSize, blockSize>>>(
            d_partitionOutputs_[i], input, startIndices[i], endIndices[i], smId
        );
        
        // Update statistics
        smStats_[smId].executionCount++;
    }
    
    cudaDeviceSynchronize();
}

void SpatialRedundancyValidator::executeReplicatedSM(
    const void* input, size_t numElements, const KernelConfig& config) {
    
    // Execute full computation on each selected SM
    for (size_t i = 0; i < selectedSMs_.size(); i++) {
        int smId = selectedSMs_[i];
        
        // Create SM-specific config
        KernelConfig smConfig = config;
        
        // Execute kernel
        kernelFunc_(d_partitionOutputs_[i], input, numElements, smConfig, smId);
        
        // Update statistics
        smStats_[smId].executionCount++;
    }
    
    cudaDeviceSynchronize();
}

void SpatialRedundancyValidator::executeCheckerboard(
    const void* input, size_t numElements, const KernelConfig& config) {
    
    // Implement checkerboard pattern distribution
    // This is a simplified version - real implementation would distribute
    // work in a checkerboard pattern across SMs
    executePartitionedSM(input, numElements, config);
}

void SpatialRedundancyValidator::executeRoundRobin(
    const void* input, size_t numElements, const KernelConfig& config) {
    
    // Implement round-robin distribution
    // This is a simplified version
    executePartitionedSM(input, numElements, config);
}

void SpatialRedundancyValidator::partitionWork(
    size_t numElements, std::vector<size_t>& startIndices, 
    std::vector<size_t>& endIndices) {
    
    startIndices.clear();
    endIndices.clear();
    
    size_t elementsPerSM = numElements / selectedSMs_.size();
    size_t remainder = numElements % selectedSMs_.size();
    
    size_t currentStart = 0;
    for (size_t i = 0; i < selectedSMs_.size(); i++) {
        size_t partitionSize = elementsPerSM;
        if (i < remainder) {
            partitionSize++;
        }
        
        startIndices.push_back(currentStart);
        endIndices.push_back(currentStart + partitionSize);
        currentStart += partitionSize;
    }
}

void SpatialRedundancyValidator::mergePartitionedResults(
    void* output, size_t numElements, size_t elementSize) {
    
    // Prepare partition info
    std::vector<size_t> partitionSizes;
    std::vector<size_t> startIndices, endIndices;
    partitionWork(numElements, startIndices, endIndices);
    
    for (size_t i = 0; i < selectedSMs_.size(); i++) {
        partitionSizes.push_back(endIndices[i] - startIndices[i]);
    }
    
    // Copy partition info to device
    size_t* d_partitionSizes;
    void** d_partitionPtrs;
    cudaMalloc(&d_partitionSizes, partitionSizes.size() * sizeof(size_t));
    cudaMalloc(&d_partitionPtrs, d_partitionOutputs_.size() * sizeof(void*));
    
    cudaMemcpy(d_partitionSizes, partitionSizes.data(), 
              partitionSizes.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_partitionPtrs, d_partitionOutputs_.data(),
              d_partitionOutputs_.size() * sizeof(void*), cudaMemcpyHostToDevice);
    
    // Launch merge kernel
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    
    mergePartitionsKernel<<<gridSize, blockSize>>>(
        output, d_partitionPtrs, d_partitionSizes, 
        selectedSMs_.size(), numElements
    );
    
    cudaFree(d_partitionSizes);
    cudaFree(d_partitionPtrs);
    cudaDeviceSynchronize();
}

bool SpatialRedundancyValidator::compareReplicatedResults(
    size_t numElements, size_t elementSize) {
    
    if (selectedSMs_.size() < 2) {
        return true;  // Nothing to compare
    }
    
    // Compare all replicas
    bool h_hasErrors = false;
    bool* d_hasErrors;
    cudaMalloc(&d_hasErrors, sizeof(bool));
    cudaMemcpy(d_hasErrors, &h_hasErrors, sizeof(bool), cudaMemcpyHostToDevice);
    
    // Prepare partition pointers
    void** d_partitionPtrs;
    cudaMalloc(&d_partitionPtrs, d_partitionOutputs_.size() * sizeof(void*));
    cudaMemcpy(d_partitionPtrs, d_partitionOutputs_.data(),
              d_partitionOutputs_.size() * sizeof(void*), cudaMemcpyHostToDevice);
    
    // Launch comparison kernel
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    
    comparePartitionsKernel<<<gridSize, blockSize>>>(
        d_partitionPtrs, numElements, selectedSMs_.size(), d_hasErrors
    );
    
    cudaMemcpy(&h_hasErrors, d_hasErrors, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_hasErrors);
    cudaFree(d_partitionPtrs);
    
    // If no errors, copy first result as output
    if (!h_hasErrors) {
        cudaMemcpy(d_mergedOutput_, d_partitionOutputs_[0], 
                  numElements * elementSize, cudaMemcpyDeviceToDevice);
    }
    
    return !h_hasErrors;
}

void SpatialRedundancyValidator::setThreadSMAffinity(int smId) {
    // This would require CUDA runtime modifications
    // Simplified version - does not actually set affinity
}

int SpatialRedundancyValidator::getCurrentSM() {
    // This would need to be called from device code
    return 0;
}