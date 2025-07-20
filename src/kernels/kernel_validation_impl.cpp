#include "kernel_interface.h"
#include "validation_engine.h"
#include <iostream>

// Template specialization for validation implementation
namespace KernelValidation {
    
    void performValidation(
        ValidationEngine* validationEngine,
        const void* outputData,
        size_t outputElements,
        size_t elementSize,
        KernelResult& result,
        const KernelConfig& config) {
        
        if (!validationEngine || !outputData) {
            return;
        }
        
        // Inject SDC if requested
        if (config.injectSDC) {
            validationEngine->injectSDC(
                const_cast<void*>(outputData),
                outputElements,
                elementSize
            );
        }
        
        try {
            // Call the validation engine to validate the output data
            auto validationResults = validationEngine->validate(
                outputData,
                outputElements,
                elementSize,
                config
            );
        
        // Process validation results
        result.validationPerformed = true;
        result.validationResults = validationResults;
        
        // Count SDCs and validation overhead
        double totalValidationTime = 0.0;
        for (const auto& valResult : validationResults) {
            if (!valResult.passed) {
                result.sdcDetectedCount += valResult.corruptedElements;
            }
            totalValidationTime += valResult.validationTimeMs;
        }
        result.validationOverheadMs = totalValidationTime;
        
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Validation failed with exception: " << e.what() << std::endl;
            result.validationPerformed = false;
        } catch (...) {
            std::cerr << "[ERROR] Validation failed with unknown exception" << std::endl;
            result.validationPerformed = false;
        }
    }
}