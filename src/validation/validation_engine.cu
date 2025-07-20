#include "validation_engine.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <thread>
#include <cstring>
#include "monitoring_engine.h"
#include "data_store.h"

// Kernel for bit flip injection
__global__ void bitFlipKernel(uint8_t* data, size_t numBytes, 
                             curandState* states, double probability) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBytes) return;
    
    curandState localState = states[tid % gridDim.x];
    float rand = curand_uniform(&localState);
    
    if (rand < probability) {
        // Flip a random bit in this byte
        int bitPos = curand(&localState) % 8;
        data[tid] ^= (1 << bitPos);
    }
    
    states[tid % gridDim.x] = localState;
}

// Kernel for memory pattern corruption
__global__ void memoryPatternKernel(uint8_t* data, size_t numBytes,
                                   curandState* states, double probability) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numBytes) return;
    
    curandState localState = states[tid % gridDim.x];
    float rand = curand_uniform(&localState);
    
    if (rand < probability) {
        // Corrupt with common memory patterns
        int pattern = curand(&localState) % 4;
        switch (pattern) {
            case 0: data[tid] = 0xFF; break;  // All ones
            case 1: data[tid] = 0x00; break;  // All zeros
            case 2: data[tid] = 0xAA; break;  // Alternating 1
            case 3: data[tid] = 0x55; break;  // Alternating 2
        }
    }
    
    states[tid % gridDim.x] = localState;
}

// Kernel to initialize random states
__global__ void initRandStates(curandState* states, int numStates, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates) return;
    
    curand_init(seed, tid, 0, &states[tid]);
}

ValidationEngine::ValidationEngine()
    : d_randStates_(nullptr)
    , numRandStates_(0)
    , monitoringEngine_(nullptr) {
}

ValidationEngine::~ValidationEngine() {
    cleanupRandom();
    for (auto& method : methods_) {
        method->cleanup();
    }
}

void ValidationEngine::initialize(const ValidationConfig& config) {
    config_ = config;
    initializeRandom();
}

void ValidationEngine::registerMethod(std::unique_ptr<ValidationMethod> method) {
    methods_.push_back(std::move(method));
}

std::vector<ValidationResult> ValidationEngine::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& kernelConfig) {
    
    std::vector<ValidationResult> results;
    
    for (const auto& method : methods_) {
        if (isValidationEnabled(config_.validationTypes, method->getType())) {
            // Setup the method if not already done
            method->setup(kernelConfig);
            
            auto start = std::chrono::high_resolution_clock::now();
            
            ValidationResult result = method->validate(data, numElements, elementSize, kernelConfig);
            
            auto end = std::chrono::high_resolution_clock::now();
            result.validationTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
            
            results.push_back(result);
            
            // Report to monitoring engine if available
            if (monitoringEngine_) {
                monitoringEngine_->recordValidationResult(result);
            }
            
            // Update statistics
            stats_.totalChecks++;
            if (!result.passed) {
                stats_.failedChecks++;
                stats_.failuresByMethod[result.method]++;
                if (result.corruptedElements > 0) {
                    stats_.detectedErrors += result.corruptedElements;
                }
            }
        }
    }
    
    stats_.update();
    return results;
}

void ValidationEngine::injectSDC(void* data, size_t numElements, size_t elementSize) {
    if (config_.injectionType == SDCInjectionType::NONE || !shouldInjectSDC()) {
        return;
    }
    
    // size_t numBytes = numElements * elementSize;  // TODO: Will be used when expanding injection types
    
    switch (config_.injectionType) {
        case SDCInjectionType::BITFLIP:
            injectBitFlips(data, numElements, elementSize);
            break;
        case SDCInjectionType::MEMORY_PATTERN:
            injectMemoryPattern(data, numElements, elementSize);
            break;
        case SDCInjectionType::TIMING:
            injectTimingErrors(data, numElements, elementSize);
            break;
        case SDCInjectionType::THERMAL:
            injectThermalErrors(data, numElements, elementSize);
            break;
        default:
            break;
    }
    
    stats_.injectedErrors++;
    stats_.injectionsByType[config_.injectionType]++;
    
    // Report injection to monitoring if available
    if (monitoringEngine_ && monitoringEngine_->getDataStore()) {
        auto* dataStore = monitoringEngine_->getDataStore();
        dataStore->insert("sdc.injected_count", 1.0);
        dataStore->insert("sdc.injection_probability", config_.sdcProbability);
    }
}

void ValidationEngine::enableValidation(ValidationType type) {
    config_.validationTypes = config_.validationTypes | type;
}

void ValidationEngine::disableValidation(ValidationType type) {
    config_.validationTypes = static_cast<ValidationType>(
        static_cast<int>(config_.validationTypes) & ~static_cast<int>(type));
}

void ValidationEngine::setSDCInjection(SDCInjectionType type, double probability) {
    config_.injectionType = type;
    config_.sdcProbability = probability;
}

void ValidationEngine::initializeRandom() {
    if (d_randStates_ != nullptr) {
        return;
    }
    
    // Initialize random states for SDC injection
    numRandStates_ = 1024;  // Number of random states
    cudaMalloc(&d_randStates_, numRandStates_ * sizeof(curandState));
    
    int blockSize = 256;
    int gridSize = (numRandStates_ + blockSize - 1) / blockSize;
    
    initRandStates<<<gridSize, blockSize>>>(d_randStates_, numRandStates_, config_.randomSeed);
    cudaDeviceSynchronize();
}

void ValidationEngine::cleanupRandom() {
    if (d_randStates_ != nullptr) {
        cudaFree(d_randStates_);
        d_randStates_ = nullptr;
        numRandStates_ = 0;
    }
}

bool ValidationEngine::shouldInjectSDC() const {
    static std::mt19937 gen(config_.randomSeed);
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen) < config_.sdcProbability;
}

void ValidationEngine::injectBitFlips(void* data, size_t numElements, size_t elementSize) {
    size_t numBytes = numElements * elementSize;
    uint8_t* byteData = static_cast<uint8_t*>(data);
    
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    
    bitFlipKernel<<<gridSize, blockSize>>>(byteData, numBytes, d_randStates_, config_.sdcProbability);
    cudaDeviceSynchronize();
}

void ValidationEngine::injectMemoryPattern(void* data, size_t numElements, size_t elementSize) {
    size_t numBytes = numElements * elementSize;
    uint8_t* byteData = static_cast<uint8_t*>(data);
    
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    
    memoryPatternKernel<<<gridSize, blockSize>>>(byteData, numBytes, d_randStates_, config_.sdcProbability);
    cudaDeviceSynchronize();
}

void ValidationEngine::injectTimingErrors(void* data, size_t numElements, size_t elementSize) {
    // Simulate timing errors by introducing delays and potential race conditions
    // This is a simplified implementation - real timing errors are more complex
    
    // Random delay to simulate timing variations
    static std::mt19937 gen(config_.randomSeed);
    static std::uniform_int_distribution<> dis(0, 1000);
    
    int delay_us = dis(gen);
    if (delay_us > 900) {  // 10% chance of significant delay
        std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
        
        // After delay, corrupt some data as if a race condition occurred
        injectBitFlips(data, numElements, elementSize);
    }
}

void ValidationEngine::injectThermalErrors(void* data, size_t numElements, size_t elementSize) {
    // Simulate thermal-induced errors
    // In reality, these would be correlated with temperature and power consumption
    
    // For simulation, we increase error probability based on a "temperature" factor
    double originalProb = config_.sdcProbability;
    
    // Simulate temperature effect (simplified model)
    static std::mt19937 gen(config_.randomSeed);
    static std::normal_distribution<> tempDist(70.0, 10.0);  // Mean 70C, std 10C
    
    double simulatedTemp = tempDist(gen);
    if (simulatedTemp > 80.0) {
        // Higher temperature increases error probability exponentially
        double tempFactor = std::exp((simulatedTemp - 80.0) / 10.0);
        config_.sdcProbability = std::min(1.0, originalProb * tempFactor);
    }
    
    // Inject errors with temperature-adjusted probability
    injectMemoryPattern(data, numElements, elementSize);
    
    // Restore original probability
    config_.sdcProbability = originalProb;
}