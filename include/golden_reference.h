#ifndef GOLDEN_REFERENCE_H
#define GOLDEN_REFERENCE_H

#include "validation_engine.h"
#include <vector>
#include <memory>
#include <functional>

// Golden reference validation method
// Compares GPU results against CPU-computed reference values
class GoldenReferenceValidator : public ValidationMethod {
public:
    GoldenReferenceValidator();
    ~GoldenReferenceValidator() override;
    
    // ValidationMethod interface
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override;
    
    std::string getName() const override { return "Golden Reference"; }
    ValidationType getType() const override { return ValidationType::GOLDEN_REFERENCE; }
    
    void setup(const KernelConfig& config) override;
    void cleanup() override;
    
    // Set the kernel type for appropriate reference computation
    enum class KernelType {
        MATRIX_MULTIPLY,
        VECTOR_ADD,
        REDUCTION,
        CONVOLUTION,
        CUSTOM
    };
    
    void setKernelType(KernelType type) { kernelType_ = type; }
    
    // Set custom reference computation function
    using ReferenceFunction = std::function<void(void* output, const void* input, size_t numElements)>;
    void setCustomReference(ReferenceFunction func) { customReference_ = func; }
    
    // Set tolerance for floating-point comparisons
    void setTolerance(double relativeTol, double absoluteTol) {
        relativeTolerance_ = relativeTol;
        absoluteTolerance_ = absoluteTol;
    }
    
private:
    KernelType kernelType_;
    ReferenceFunction customReference_;
    
    // Tolerance for floating-point comparisons
    double relativeTolerance_;
    double absoluteTolerance_;
    
    // Reference data storage
    std::vector<uint8_t> referenceData_;
    
    // Reference computation methods
    void computeMatrixMultiplyReference(void* output, size_t matrixSize);
    void computeVectorAddReference(void* output, size_t numElements);
    void computeReductionReference(void* output, size_t numElements);
    void computeConvolutionReference(void* output, size_t numElements);
    
    // Comparison methods for different data types
    template<typename T>
    ValidationResult compareData(const T* gpuData, const T* cpuData, size_t numElements);
    
    ValidationResult compareFloatingPoint(const void* gpuData, const void* cpuData, 
                                        size_t numElements, size_t elementSize);
    ValidationResult compareInteger(const void* gpuData, const void* cpuData,
                                  size_t numElements, size_t elementSize);
};

// Helper functions for reference computations
namespace GoldenReferenceHelpers {
    // Matrix multiply reference (C = A * B)
    template<typename T>
    void matrixMultiplyCPU(T* C, const T* A, const T* B, size_t N);
    
    // Vector addition reference (C = A + B)
    template<typename T>
    void vectorAddCPU(T* C, const T* A, const T* B, size_t N);
    
    // Reduction reference (sum of all elements)
    template<typename T>
    T reductionCPU(const T* data, size_t N);
    
    // Simple 2D convolution reference
    template<typename T>
    void convolution2DCPU(T* output, const T* input, const T* kernel,
                         size_t width, size_t height, size_t kernelSize);
}

#endif // GOLDEN_REFERENCE_H