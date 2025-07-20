#include "tmr_validator.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

// GPU kernel for majority voting
__global__ void majorityVoteKernel(const uint8_t* in1, const uint8_t* in2, 
                                  const uint8_t* in3, uint8_t* out, size_t numBytes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBytes) return;
    
    uint8_t byte1 = in1[tid];
    uint8_t byte2 = in2[tid];
    uint8_t byte3 = in3[tid];
    
    // Majority vote on each byte
    if (byte1 == byte2 || byte1 == byte3) {
        out[tid] = byte1;
    } else {
        out[tid] = byte2;  // byte2 == byte3 in this case
    }
}

// GPU kernel for bitwise voting
__global__ void bitwiseVoteKernel(const uint8_t* in1, const uint8_t* in2,
                                 const uint8_t* in3, uint8_t* out, size_t numBytes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBytes) return;
    
    uint8_t byte1 = in1[tid];
    uint8_t byte2 = in2[tid];
    uint8_t byte3 = in3[tid];
    uint8_t result = 0;
    
    // Vote on each bit position
    for (int bit = 0; bit < 8; ++bit) {
        int bit1 = (byte1 >> bit) & 1;
        int bit2 = (byte2 >> bit) & 1;
        int bit3 = (byte3 >> bit) & 1;
        
        // Majority vote
        int votedBit = (bit1 + bit2 + bit3) >= 2 ? 1 : 0;
        result |= (votedBit << bit);
    }
    
    out[tid] = result;
}

// GPU kernel for comparing and counting differences
__global__ void compareAndCountKernel(const uint8_t* data1, const uint8_t* data2,
                                    size_t numBytes, uint32_t* diffCount) {
    extern __shared__ uint32_t sharedCount[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint32_t localCount = 0;
    
    for (size_t i = gid; i < numBytes; i += blockDim.x * gridDim.x) {
        if (data1[i] != data2[i]) {
            localCount++;
        }
    }
    
    sharedCount[tid] = localCount;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedCount[tid] += sharedCount[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(diffCount, sharedCount[0]);
    }
}

// ConcurrentTMRExecutor implementation
ConcurrentTMRExecutor::ConcurrentTMRExecutor() {
    cudaStreamCreate(&stream1_);
    cudaStreamCreate(&stream2_);
    cudaStreamCreate(&stream3_);
}

ConcurrentTMRExecutor::~ConcurrentTMRExecutor() {
    cudaStreamDestroy(stream1_);
    cudaStreamDestroy(stream2_);
    cudaStreamDestroy(stream3_);
}

void ConcurrentTMRExecutor::execute(void* out1, void* out2, void* out3,
                                   const void* input, size_t numElements,
                                   const TMRValidator::KernelFunction& kernelFunc,
                                   const KernelConfig& config) {
    // Create modified configs for each stream
    KernelConfig config1 = config;
    KernelConfig config2 = config;
    KernelConfig config3 = config;
    
    // Execute kernels concurrently on different streams
    cudaStreamSynchronize(stream1_);
    cudaStreamSynchronize(stream2_);
    cudaStreamSynchronize(stream3_);
    
    kernelFunc(out1, input, numElements, config1);
    kernelFunc(out2, input, numElements, config2);
    kernelFunc(out3, input, numElements, config3);
    
    // Synchronize all streams
    cudaStreamSynchronize(stream1_);
    cudaStreamSynchronize(stream2_);
    cudaStreamSynchronize(stream3_);
}

// TMRValidator implementation
TMRValidator::TMRValidator(VotingStrategy strategy, TMRMode mode)
    : votingStrategy_(strategy)
    , tmrMode_(mode)
    , diagnosticsEnabled_(false)
    , d_output1_(nullptr)
    , d_output2_(nullptr)
    , d_output3_(nullptr)
    , d_votedOutput_(nullptr)
    , bufferSize_(0) {
}

TMRValidator::~TMRValidator() {
    cleanup();
}

void TMRValidator::setup(const KernelConfig& config) {
    // Calculate buffer size based on expected output
    size_t requiredSize = config.matrixSize * config.matrixSize * sizeof(float);
    
    if (bufferSize_ < requiredSize) {
        cleanup();  // Free existing buffers
        
        cudaMalloc(&d_output1_, requiredSize);
        cudaMalloc(&d_output2_, requiredSize);
        cudaMalloc(&d_output3_, requiredSize);
        cudaMalloc(&d_votedOutput_, requiredSize);
        
        bufferSize_ = requiredSize;
    }
}

void TMRValidator::cleanup() {
    if (d_output1_) {
        cudaFree(d_output1_);
        d_output1_ = nullptr;
    }
    if (d_output2_) {
        cudaFree(d_output2_);
        d_output2_ = nullptr;
    }
    if (d_output3_) {
        cudaFree(d_output3_);
        d_output3_ = nullptr;
    }
    if (d_votedOutput_) {
        cudaFree(d_votedOutput_);
        d_votedOutput_ = nullptr;
    }
    bufferSize_ = 0;
}

ValidationResult TMRValidator::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& config) {
    
    ValidationResult result;
    result.method = ValidationType::TMR;
    
    if (!kernelFunc_) {
        result.passed = false;
        result.errorDetails = "No kernel function set for TMR validation";
        return result;
    }
    
    // Execute TMR based on mode
    switch (tmrMode_) {
        case TMRMode::SEQUENTIAL:
            executeSequential(data, numElements, config);
            break;
        case TMRMode::CONCURRENT:
            executeConcurrent(data, numElements, config);
            break;
        case TMRMode::MULTI_GPU:
            executeMultiGPU(data, numElements, config);
            break;
    }
    
    stats_.totalRuns++;
    
    // Perform voting based on strategy
    size_t numBytes = numElements * elementSize;
    
    switch (votingStrategy_) {
        case VotingStrategy::MAJORITY:
            result = performMajorityVoting(d_output1_, d_output2_, d_output3_,
                                         d_votedOutput_, numElements, elementSize);
            break;
        case VotingStrategy::UNANIMOUS:
            result = performUnanimousVoting(d_output1_, d_output2_, d_output3_,
                                          d_votedOutput_, numElements, elementSize);
            break;
        case VotingStrategy::WEIGHTED:
            result = performWeightedVoting(d_output1_, d_output2_, d_output3_,
                                         d_votedOutput_, numElements, elementSize);
            break;
        case VotingStrategy::BITWISE:
            result = performBitwiseVoting(d_output1_, d_output2_, d_output3_,
                                        d_votedOutput_, numElements, elementSize);
            break;
    }
    
    // Copy voted output back to original data location
    if (result.passed) {
        cudaMemcpy(const_cast<void*>(data), d_votedOutput_, numBytes, cudaMemcpyDeviceToDevice);
    }
    
    // Update statistics
    if (!result.passed) {
        stats_.consensusFailures++;
    }
    
    if (diagnosticsEnabled_) {
        analyzeDifferences(d_output1_, d_output2_, d_output3_, numElements, elementSize);
    }
    
    return result;
}

void TMRValidator::executeSequential(const void* input, size_t numElements, const KernelConfig& config) {
    // Execute kernel 3 times sequentially
    kernelFunc_(d_output1_, input, numElements, config);
    cudaDeviceSynchronize();
    
    kernelFunc_(d_output2_, input, numElements, config);
    cudaDeviceSynchronize();
    
    kernelFunc_(d_output3_, input, numElements, config);
    cudaDeviceSynchronize();
}

void TMRValidator::executeConcurrent(const void* input, size_t numElements, const KernelConfig& config) {
    ConcurrentTMRExecutor executor;
    executor.execute(d_output1_, d_output2_, d_output3_, input, numElements, kernelFunc_, config);
}

void TMRValidator::executeMultiGPU(const void* input, size_t numElements, const KernelConfig& config) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount < 3) {
        // Fall back to sequential execution
        executeSequential(input, numElements, config);
        return;
    }
    
    // Execute on different GPUs
    int originalDevice;
    cudaGetDevice(&originalDevice);
    
    // GPU 0
    cudaSetDevice(0);
    kernelFunc_(d_output1_, input, numElements, config);
    
    // GPU 1
    cudaSetDevice(1);
    void* d_input_gpu1;
    void* d_output_gpu1;
    size_t dataSize = numElements * sizeof(float);  // Assuming float data
    cudaMalloc(&d_input_gpu1, dataSize);
    cudaMalloc(&d_output_gpu1, bufferSize_);
    cudaMemcpy(d_input_gpu1, input, dataSize, cudaMemcpyDeviceToDevice);
    kernelFunc_(d_output_gpu1, d_input_gpu1, numElements, config);
    
    // GPU 2
    cudaSetDevice(2);
    void* d_input_gpu2;
    void* d_output_gpu2;
    cudaMalloc(&d_input_gpu2, dataSize);
    cudaMalloc(&d_output_gpu2, bufferSize_);
    cudaMemcpy(d_input_gpu2, input, dataSize, cudaMemcpyDeviceToDevice);
    kernelFunc_(d_output_gpu2, d_input_gpu2, numElements, config);
    
    // Copy results back to original GPU
    cudaSetDevice(originalDevice);
    cudaMemcpy(d_output2_, d_output_gpu1, bufferSize_, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_output3_, d_output_gpu2, bufferSize_, cudaMemcpyDeviceToDevice);
    
    // Cleanup
    cudaSetDevice(1);
    cudaFree(d_input_gpu1);
    cudaFree(d_output_gpu1);
    
    cudaSetDevice(2);
    cudaFree(d_input_gpu2);
    cudaFree(d_output_gpu2);
    
    cudaSetDevice(originalDevice);
}

ValidationResult TMRValidator::performMajorityVoting(
    const void* out1, const void* out2, const void* out3, void* votedOutput,
    size_t numElements, size_t elementSize) {
    
    ValidationResult result;
    result.method = ValidationType::TMR;
    
    size_t numBytes = numElements * elementSize;
    
    // Launch majority vote kernel
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    
    majorityVoteKernel<<<gridSize, blockSize>>>(
        static_cast<const uint8_t*>(out1),
        static_cast<const uint8_t*>(out2),
        static_cast<const uint8_t*>(out3),
        static_cast<uint8_t*>(votedOutput),
        numBytes
    );
    cudaDeviceSynchronize();
    
    // Check if all outputs agree
    bool allAgree = compareOutputs(out1, out2, numBytes) &&
                   compareOutputs(out2, out3, numBytes);
    
    if (allAgree) {
        result.passed = true;
        result.confidence = 1.0;
    } else {
        // Check pairwise agreement
        bool out1_2_agree = compareOutputs(out1, out2, numBytes);
        bool out1_3_agree = compareOutputs(out1, out3, numBytes);
        bool out2_3_agree = compareOutputs(out2, out3, numBytes);
        
        if (out1_2_agree || out1_3_agree || out2_3_agree) {
            result.passed = true;
            result.confidence = 0.67;  // 2 out of 3 agree
            result.errorDetails = "TMR detected disagreement, majority vote applied";
        } else {
            result.passed = false;
            result.confidence = 0.0;
            result.errorDetails = "No consensus in TMR outputs";
            stats_.consensusFailures++;
        }
    }
    
    return result;
}

ValidationResult TMRValidator::performUnanimousVoting(
    const void* out1, const void* out2, const void* out3, void* votedOutput,
    size_t numElements, size_t elementSize) {
    
    ValidationResult result;
    result.method = ValidationType::TMR;
    
    size_t numBytes = numElements * elementSize;
    
    // Check if all outputs agree
    bool allAgree = compareOutputs(out1, out2, numBytes) &&
                   compareOutputs(out2, out3, numBytes);
    
    if (allAgree) {
        // Copy any output as voted result
        cudaMemcpy(votedOutput, out1, numBytes, cudaMemcpyDeviceToDevice);
        result.passed = true;
        result.confidence = 1.0;
    } else {
        result.passed = false;
        result.confidence = 0.0;
        result.errorDetails = "TMR unanimous voting failed - outputs disagree";
        
        // Count differences for diagnostics
        size_t diff_1_2 = countDifferences(out1, out2, numBytes);
        size_t diff_1_3 = countDifferences(out1, out3, numBytes);
        size_t diff_2_3 = countDifferences(out2, out3, numBytes);
        
        result.errorDetails += " (Differences: 1-2=" + std::to_string(diff_1_2) +
                             ", 1-3=" + std::to_string(diff_1_3) +
                             ", 2-3=" + std::to_string(diff_2_3) + ")";
    }
    
    return result;
}

ValidationResult TMRValidator::performWeightedVoting(
    const void* out1, const void* out2, const void* out3, void* votedOutput,
    size_t numElements, size_t elementSize) {
    
    // For now, implement as majority voting
    // Future: implement confidence-based weighted voting
    return performMajorityVoting(out1, out2, out3, votedOutput, numElements, elementSize);
}

ValidationResult TMRValidator::performBitwiseVoting(
    const void* out1, const void* out2, const void* out3, void* votedOutput,
    size_t numElements, size_t elementSize) {
    
    ValidationResult result;
    result.method = ValidationType::TMR;
    
    size_t numBytes = numElements * elementSize;
    
    // Launch bitwise vote kernel
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    
    bitwiseVoteKernel<<<gridSize, blockSize>>>(
        static_cast<const uint8_t*>(out1),
        static_cast<const uint8_t*>(out2),
        static_cast<const uint8_t*>(out3),
        static_cast<uint8_t*>(votedOutput),
        numBytes
    );
    cudaDeviceSynchronize();
    
    // Count bit differences
    size_t totalBitDiffs = 0;
    size_t diff_1_voted = countDifferences(out1, votedOutput, numBytes);
    size_t diff_2_voted = countDifferences(out2, votedOutput, numBytes);
    size_t diff_3_voted = countDifferences(out3, votedOutput, numBytes);
    
    totalBitDiffs = diff_1_voted + diff_2_voted + diff_3_voted;
    
    if (totalBitDiffs == 0) {
        result.passed = true;
        result.confidence = 1.0;
    } else {
        result.passed = true;  // Bitwise voting always produces a result
        result.confidence = 1.0 - (static_cast<double>(totalBitDiffs) / (3 * numBytes * 8));
        result.errorDetails = "Bitwise TMR corrected " + std::to_string(totalBitDiffs) + " bit errors";
        
        if (totalBitDiffs == 1) {
            stats_.singleBitErrors++;
        } else {
            stats_.multiBitErrors++;
        }
    }
    
    return result;
}

bool TMRValidator::compareOutputs(const void* out1, const void* out2, size_t numBytes) {
    // Use CUDA kernel to compare
    uint32_t h_diffCount = 0;
    uint32_t* d_diffCount;
    cudaMalloc(&d_diffCount, sizeof(uint32_t));
    cudaMemcpy(d_diffCount, &h_diffCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(uint32_t);
    
    compareAndCountKernel<<<gridSize, blockSize, sharedMemSize>>>(
        static_cast<const uint8_t*>(out1),
        static_cast<const uint8_t*>(out2),
        numBytes, d_diffCount
    );
    
    cudaMemcpy(&h_diffCount, d_diffCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_diffCount);
    
    return h_diffCount == 0;
}

size_t TMRValidator::countDifferences(const void* out1, const void* out2, size_t numBytes) {
    uint32_t h_diffCount = 0;
    uint32_t* d_diffCount;
    cudaMalloc(&d_diffCount, sizeof(uint32_t));
    cudaMemcpy(d_diffCount, &h_diffCount, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(uint32_t);
    
    compareAndCountKernel<<<gridSize, blockSize, sharedMemSize>>>(
        static_cast<const uint8_t*>(out1),
        static_cast<const uint8_t*>(out2),
        numBytes, d_diffCount
    );
    
    cudaMemcpy(&h_diffCount, d_diffCount, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_diffCount);
    
    return static_cast<size_t>(h_diffCount);
}

void TMRValidator::analyzeDifferences(const void* out1, const void* out2, const void* out3,
                                    size_t numElements, size_t elementSize) {
    size_t numBytes = numElements * elementSize;
    
    size_t diff_1_2 = countDifferences(out1, out2, numBytes);
    size_t diff_1_3 = countDifferences(out1, out3, numBytes);
    size_t diff_2_3 = countDifferences(out2, out3, numBytes);
    
    std::cout << "TMR Diagnostic Analysis:" << std::endl;
    std::cout << "  Differences between output 1 and 2: " << diff_1_2 << " bytes" << std::endl;
    std::cout << "  Differences between output 1 and 3: " << diff_1_3 << " bytes" << std::endl;
    std::cout << "  Differences between output 2 and 3: " << diff_2_3 << " bytes" << std::endl;
    
    // Update statistics
    double agreementRate = 1.0 - (static_cast<double>(diff_1_2 + diff_1_3 + diff_2_3) / (3 * numBytes));
    stats_.averageAgreementRate = (stats_.averageAgreementRate * (stats_.totalRuns - 1) + agreementRate) / stats_.totalRuns;
}