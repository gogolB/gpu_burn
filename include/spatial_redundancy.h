#ifndef SPATIAL_REDUNDANCY_H
#define SPATIAL_REDUNDANCY_H

#include "validation_engine.h"
#include <vector>
#include <functional>
#include <cuda_runtime.h>

// Spatial distribution strategies
enum class SpatialStrategy {
    PARTITION_SM,        // Partition work across different SMs
    REPLICATE_SM,       // Replicate same work on different SMs
    CHECKERBOARD,       // Checkerboard pattern distribution
    ROUND_ROBIN         // Round-robin distribution across SMs
};

// SM selection modes
enum class SMSelectionMode {
    ALL_SMS,            // Use all available SMs
    ALTERNATE_SMS,      // Use alternating SMs
    RANDOM_SMS,         // Randomly select SMs
    MANUAL_SMS          // Manually specify SM mask
};

// Spatial Redundancy validator
// Distributes computation across different SMs to detect location-specific errors
class SpatialRedundancyValidator : public ValidationMethod {
public:
    SpatialRedundancyValidator(SpatialStrategy strategy = SpatialStrategy::PARTITION_SM,
                               SMSelectionMode smMode = SMSelectionMode::ALL_SMS);
    ~SpatialRedundancyValidator() override;
    
    // ValidationMethod interface
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override;
    
    std::string getName() const override { return "Spatial Redundancy"; }
    ValidationType getType() const override { return ValidationType::SPATIAL_REDUNDANCY; }
    
    void setup(const KernelConfig& config) override;
    void cleanup() override;
    
    // Set kernel execution function
    using KernelFunction = std::function<void(void* output, const void* input, 
                                             size_t numElements, const KernelConfig& config,
                                             int smId)>;
    void setKernelFunction(KernelFunction func) { kernelFunc_ = func; }
    
    // Configuration methods
    void setSpatialStrategy(SpatialStrategy strategy) { strategy_ = strategy; }
    void setSMSelectionMode(SMSelectionMode mode) { smMode_ = mode; }
    void setSMMask(uint64_t mask) { smMask_ = mask; }
    
    // Get SM-specific statistics
    struct SMStats {
        int smId;
        size_t executionCount;
        size_t errorCount;
        double averagePerformance;
        bool hasErrors;
        
        SMStats() : smId(-1), executionCount(0), errorCount(0), 
                   averagePerformance(0.0), hasErrors(false) {}
    };
    
    std::vector<SMStats> getSMStats() const { return smStats_; }
    
private:
    SpatialStrategy strategy_;
    SMSelectionMode smMode_;
    KernelFunction kernelFunc_;
    
    // GPU properties
    int numSMs_;
    int maxThreadsPerSM_;
    cudaDeviceProp deviceProps_;
    
    // SM selection
    uint64_t smMask_;
    std::vector<int> selectedSMs_;
    
    // Device buffers for spatial execution
    std::vector<void*> d_partitionOutputs_;
    void* d_mergedOutput_;
    size_t bufferSize_;
    
    // SM-specific statistics
    std::vector<SMStats> smStats_;
    
    // Execution methods
    void executePartitionedSM(const void* input, size_t numElements, const KernelConfig& config);
    void executeReplicatedSM(const void* input, size_t numElements, const KernelConfig& config);
    void executeCheckerboard(const void* input, size_t numElements, const KernelConfig& config);
    void executeRoundRobin(const void* input, size_t numElements, const KernelConfig& config);
    
    // Helper methods
    void selectSMs();
    void partitionWork(size_t numElements, std::vector<size_t>& startIndices, 
                      std::vector<size_t>& endIndices);
    void mergePartitionedResults(void* output, size_t numElements, size_t elementSize);
    bool compareReplicatedResults(size_t numElements, size_t elementSize);
    
    // SM affinity control
    void setThreadSMAffinity(int smId);
    int getCurrentSM();
};

// GPU kernels for spatial redundancy
__global__ void spatialExecutionKernel(void* output, const void* input, 
                                      size_t startIdx, size_t endIdx,
                                      int targetSM);

__global__ void mergePartitionsKernel(void* output, void** partitions,
                                     size_t* partitionSizes, int numPartitions,
                                     size_t totalElements);

__global__ void comparePartitionsKernel(void** partitions, size_t numElements,
                                       int numPartitions, bool* hasErrors);

// Helper functions for SM management
__device__ int getSMID();
__device__ void forceSMExecution(int targetSM);

// Utility class for SM profiling
class SMProfiler {
public:
    SMProfiler();
    ~SMProfiler();
    
    void startProfiling(int smId);
    void endProfiling(int smId);
    
    double getAverageTime(int smId) const;
    size_t getExecutionCount(int smId) const;
    
private:
    struct ProfileData {
        cudaEvent_t startEvent;
        cudaEvent_t endEvent;
        std::vector<float> executionTimes;
    };
    
    std::unordered_map<int, ProfileData> profileData_;
};

#endif // SPATIAL_REDUNDANCY_H