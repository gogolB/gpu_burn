#ifndef VALIDATION_ENGINE_H
#define VALIDATION_ENGINE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "validation_types.h"
#include "kernel_interface.h"

// Forward declarations
class ValidationMethod;
struct ValidationConfig;
class MonitoringEngine;

// Validation configuration
struct ValidationConfig {
    ValidationType validationTypes;
    SDCInjectionType injectionType;
    double sdcProbability;         // Probability of SDC injection (0.0 - 1.0)
    double validationInterval;     // Seconds between validation checks
    bool continuousValidation;     // Run validation continuously
    int randomSeed;               // Seed for reproducible SDC injection
    
    ValidationConfig()
        : validationTypes(ValidationType::NONE)
        , injectionType(SDCInjectionType::NONE)
        , sdcProbability(0.0001)  // 0.01% default
        , validationInterval(1.0)
        , continuousValidation(false)
        , randomSeed(42) {}
};

// Overall validation statistics
struct ValidationStats {
    size_t totalChecks;
    size_t failedChecks;
    size_t injectedErrors;
    size_t detectedErrors;
    double detectionRate;
    
    std::unordered_map<ValidationType, size_t> failuresByMethod;
    std::unordered_map<SDCInjectionType, size_t> injectionsByType;
    
    ValidationStats()
        : totalChecks(0)
        , failedChecks(0)
        , injectedErrors(0)
        , detectedErrors(0)
        , detectionRate(0.0) {}
    
    void update() {
        if (injectedErrors > 0) {
            detectionRate = static_cast<double>(detectedErrors) / injectedErrors;
        }
    }
};

// Base validation method interface
class ValidationMethod {
public:
    virtual ~ValidationMethod() = default;
    
    // Validate kernel output
    virtual ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) = 0;
    
    // Get method name
    virtual std::string getName() const = 0;
    virtual ValidationType getType() const = 0;
    
    // Setup method with kernel-specific parameters
    virtual void setup(const KernelConfig& config) = 0;
    
    // Cleanup resources
    virtual void cleanup() = 0;
};

// Main validation engine
class ValidationEngine {
public:
    ValidationEngine();
    ~ValidationEngine();
    
    // Initialize with configuration
    void initialize(const ValidationConfig& config);
    
    // Register validation methods
    void registerMethod(std::unique_ptr<ValidationMethod> method);
    
    // Validate kernel output
    std::vector<ValidationResult> validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& kernelConfig);
    
    // SDC injection
    void injectSDC(void* data, size_t numElements, size_t elementSize);
    
    // Get validation statistics
    ValidationStats getStats() const { return stats_; }
    void resetStats() { stats_ = ValidationStats(); }
    
    // Enable/disable specific validation methods
    void enableValidation(ValidationType type);
    void disableValidation(ValidationType type);
    
    // Set SDC injection parameters
    void setSDCInjection(SDCInjectionType type, double probability);
    
    // Monitoring integration
    void setMonitoringEngine(MonitoringEngine* engine) { monitoringEngine_ = engine; }
    MonitoringEngine* getMonitoringEngine() const { return monitoringEngine_; }
    
private:
    ValidationConfig config_;
    ValidationStats stats_;
    std::vector<std::unique_ptr<ValidationMethod>> methods_;
    
    // Random number generator for SDC injection
    curandState* d_randStates_;
    int numRandStates_;
    
    // Helper methods
    void initializeRandom();
    void cleanupRandom();
    bool shouldInjectSDC() const;
    
    // SDC injection implementations
    void injectBitFlips(void* data, size_t numElements, size_t elementSize);
    void injectMemoryPattern(void* data, size_t numElements, size_t elementSize);
    void injectTimingErrors(void* data, size_t numElements, size_t elementSize);
    void injectThermalErrors(void* data, size_t numElements, size_t elementSize);
    
    // Monitoring integration
    MonitoringEngine* monitoringEngine_;
};

#endif // VALIDATION_ENGINE_H