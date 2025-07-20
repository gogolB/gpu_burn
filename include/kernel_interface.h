#ifndef KERNEL_INTERFACE_H
#define KERNEL_INTERFACE_H

#include <cuda_runtime.h>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include "gpu_types.h"
#include "validation_types.h"

// Forward declarations
struct KernelConfig;
struct KernelResult;
class ValidationEngine;
class MonitoringEngine;

// Base interface for all GPU burn kernels
class KernelInterface {
public:
    virtual ~KernelInterface() = default;
    
    // Execute the kernel with given configuration
    virtual KernelResult execute(const KernelConfig& config) = 0;
    
    // Get kernel name and description
    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;
    
    // Check if kernel is supported on current GPU
    virtual bool isSupported(int deviceId) const = 0;
    
    // Get memory requirements
    virtual size_t getMemoryRequirement(const KernelConfig& config) const = 0;
    
    // Validation hooks
    virtual void setValidationEngine(ValidationEngine* engine) { validationEngine_ = engine; }
    virtual ValidationEngine* getValidationEngine() const { return validationEngine_; }
    
    // Enable/disable validation
    virtual void enableValidation(bool enable) { validationEnabled_ = enable; }
    virtual bool isValidationEnabled() const { return validationEnabled_; }
    
    // Get output data for validation (after kernel execution)
    virtual const void* getOutputData() const = 0;
    virtual size_t getOutputElements() const = 0;
    virtual size_t getOutputElementSize() const = 0;
    
    // Monitoring hooks
    virtual void setMonitoringEngine(MonitoringEngine* engine) { monitoringEngine_ = engine; }
    virtual MonitoringEngine* getMonitoringEngine() const { return monitoringEngine_; }
    
    // Enable/disable monitoring
    virtual void enableMonitoring(bool enable) { monitoringEnabled_ = enable; }
    virtual bool isMonitoringEnabled() const { return monitoringEnabled_; }
    
protected:
    ValidationEngine* validationEngine_ = nullptr;
    bool validationEnabled_ = false;
    MonitoringEngine* monitoringEngine_ = nullptr;
    bool monitoringEnabled_ = false;
};

// Kernel configuration parameters
struct KernelConfig {
    // Grid and block dimensions
    dim3 gridDim;
    dim3 blockDim;
    
    // Problem size
    size_t matrixSize;      // For matrix operations
    size_t numIterations;   // Number of iterations to run
    
    // Memory configuration
    size_t sharedMemSize;   // Shared memory per block
    bool useTensorCores;    // Enable tensor cores if available
    
    // Device selection
    int deviceId;
    
    // Validation configuration
    bool enableValidation;   // Enable validation for this execution
    bool injectSDC;         // Enable SDC injection
    double sdcProbability;  // Probability of SDC injection
    
    // Validation callback (called after each iteration)
    using ValidationCallback = std::function<void(int iteration, const void* data, size_t numElements)>;
    ValidationCallback validationCallback;
    
    // Monitoring callback (called after each iteration)
    using MonitoringCallback = std::function<void(int iteration, const KernelResult& result)>;
    MonitoringCallback monitoringCallback;
    
    // Default configuration
    KernelConfig()
        : gridDim(256, 1, 1)
        , blockDim(256, 1, 1)
        , matrixSize(1024)
        , numIterations(1000)
        , sharedMemSize(0)
        , useTensorCores(true)
        , deviceId(0)
        , enableValidation(false)
        , injectSDC(false)
        , sdcProbability(0.0001) {}
};

// Kernel execution results
struct KernelResult {
    bool success;
    double executionTimeMs;
    double gflops;
    size_t memoryBandwidthGBps;
    std::string errorMessage;
    
    // Performance metrics
    double avgPowerWatts;
    double avgTemperatureCelsius;
    
    // Validation results
    bool validationPerformed;
    std::vector<ValidationResult> validationResults;
    size_t sdcDetectedCount;    // Number of SDCs detected
    size_t sdcCorrectedCount;   // Number of SDCs corrected
    double validationOverheadMs; // Time spent on validation
    
    KernelResult()
        : success(false)
        , executionTimeMs(0.0)
        , gflops(0.0)
        , memoryBandwidthGBps(0.0)
        , avgPowerWatts(0.0)
        , avgTemperatureCelsius(0.0)
        , validationPerformed(false)
        , sdcDetectedCount(0)
        , sdcCorrectedCount(0)
        , validationOverheadMs(0.0) {}
};

// Namespace for validation implementation (defined in kernel_validation_impl.cpp)
namespace KernelValidation {
    void performValidation(
        ValidationEngine* validationEngine,
        const void* outputData,
        size_t outputElements,
        size_t elementSize,
        KernelResult& result,
        const KernelConfig& config);
}

// Template base class for typed kernels
template<typename T>
class TypedKernel : public KernelInterface {
protected:
    // Output data for validation
    T* d_output_ = nullptr;
    size_t outputElements_ = 0;
    
    // Allocate device memory
    T* allocateDevice(size_t elements) {
        T* ptr = nullptr;
        cudaMalloc(&ptr, elements * sizeof(T));
        return ptr;
    }
    
    // Free device memory
    void freeDevice(T* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
    
    // Copy to device
    void copyToDevice(T* dst, const T* src, size_t elements) {
        cudaMemcpy(dst, src, elements * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    // Copy from device
    void copyFromDevice(T* dst, const T* src, size_t elements) {
        cudaMemcpy(dst, src, elements * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    // Validation support
    const void* getOutputData() const override { return d_output_; }
    size_t getOutputElements() const override { return outputElements_; }
    size_t getOutputElementSize() const override { return sizeof(T); }
    
    // Perform validation if enabled
    void performValidation(KernelResult& result, const KernelConfig& config) {
        if (validationEnabled_ && validationEngine_ && d_output_) {
            // Call the external validation implementation
            KernelValidation::performValidation(
                validationEngine_,
                d_output_,
                outputElements_,
                sizeof(T),
                result,
                config
            );
        }
    }
    
    // Set output data for validation
    void setOutputData(T* output, size_t elements) {
        d_output_ = output;
        outputElements_ = elements;
    }
    
    // Perform monitoring if enabled
    void performMonitoring(const KernelResult& result, const KernelConfig& config) {
        if (monitoringEnabled_ && monitoringEngine_) {
            // Implementation would call monitoring engine
            if (config.monitoringCallback) {
                config.monitoringCallback(0, result);
            }
        }
    }
};

// Factory function type for creating kernels
using KernelFactory = std::unique_ptr<KernelInterface>(*)();

// Kernel registry macros
#define REGISTER_KERNEL(KernelClass) \
    std::unique_ptr<KernelInterface> create##KernelClass() { \
        return std::make_unique<KernelClass>(); \
    }

// Factory function declarations for new stress kernels
std::unique_ptr<KernelInterface> createPowerVirusKernel();
std::unique_ptr<KernelInterface> createThermalGradientKernel();
std::unique_ptr<KernelInterface> createMemoryControllerStressKernel();
std::unique_ptr<KernelInterface> createMixedPrecisionChaosKernel();

// Factory function declarations for LLM workload kernels
std::unique_ptr<KernelInterface> createLLMInferenceKernel();
std::unique_ptr<KernelInterface> createLLMTrainingKernel();

#endif // KERNEL_INTERFACE_H