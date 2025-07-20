#ifndef VALIDATION_TYPES_H
#define VALIDATION_TYPES_H

#include <string>
#include <memory>

// Validation types
enum class ValidationType {
    NONE = 0,
    GOLDEN_REFERENCE = 1 << 0,
    CHECKSUM = 1 << 1,
    MATHEMATICAL_INVARIANT = 1 << 2,
    TMR = 1 << 3,
    SPATIAL_REDUNDANCY = 1 << 4,
    TEMPORAL_REDUNDANCY = 1 << 5,
    CROSS_SM = 1 << 6,
    POWER_AWARE = 1 << 7,
    ALL = GOLDEN_REFERENCE | CHECKSUM | MATHEMATICAL_INVARIANT | TMR |
          SPATIAL_REDUNDANCY | TEMPORAL_REDUNDANCY | CROSS_SM | POWER_AWARE
};

// SDC injection types
enum class SDCInjectionType {
    NONE = 0,
    BITFLIP = 1,
    MEMORY_PATTERN = 2,
    TIMING = 3,
    THERMAL = 4
};

// Validation result for a single check
struct ValidationResult {
    bool passed;
    ValidationType method;
    double confidence;          // Confidence level (0.0 - 1.0)
    std::string errorDetails;
    size_t corruptedElements;   // Number of corrupted elements detected
    
    // Performance impact
    double validationTimeMs;
    double overheadPercent;
    
    ValidationResult()
        : passed(true)
        , method(ValidationType::NONE)
        , confidence(1.0)
        , corruptedElements(0)
        , validationTimeMs(0.0)
        , overheadPercent(0.0) {}
};

// Helper functions for validation types

// Get validation type name as string
inline std::string getValidationTypeName(ValidationType type) {
    switch (type) {
        case ValidationType::NONE: return "None";
        case ValidationType::GOLDEN_REFERENCE: return "Golden Reference";
        case ValidationType::CHECKSUM: return "Checksum";
        case ValidationType::MATHEMATICAL_INVARIANT: return "Mathematical Invariant";
        case ValidationType::TMR: return "Triple Modular Redundancy";
        case ValidationType::SPATIAL_REDUNDANCY: return "Spatial Redundancy";
        case ValidationType::TEMPORAL_REDUNDANCY: return "Temporal Redundancy";
        case ValidationType::CROSS_SM: return "Cross-SM Validation";
        case ValidationType::POWER_AWARE: return "Power-Aware Validation";
        case ValidationType::ALL: return "All Methods";
        default: return "Unknown";
    }
}

// Get SDC injection type name as string
inline std::string getSDCInjectionTypeName(SDCInjectionType type) {
    switch (type) {
        case SDCInjectionType::NONE: return "None";
        case SDCInjectionType::BITFLIP: return "Bit Flip";
        case SDCInjectionType::MEMORY_PATTERN: return "Memory Pattern";
        case SDCInjectionType::TIMING: return "Timing Error";
        case SDCInjectionType::THERMAL: return "Thermal Error";
        default: return "Unknown";
    }
}

// Helper to check if validation type is enabled
inline bool isValidationEnabled(ValidationType enabled, ValidationType check) {
    return (static_cast<int>(enabled) & static_cast<int>(check)) != 0;
}

// Helper to combine validation types
inline ValidationType operator|(ValidationType a, ValidationType b) {
    return static_cast<ValidationType>(static_cast<int>(a) | static_cast<int>(b));
}

inline ValidationType operator&(ValidationType a, ValidationType b) {
    return static_cast<ValidationType>(static_cast<int>(a) & static_cast<int>(b));
}


#endif // VALIDATION_TYPES_H