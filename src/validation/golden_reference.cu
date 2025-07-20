#include "golden_reference.h"
#include "validation_type_converter.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <type_traits>

GoldenReferenceValidator::GoldenReferenceValidator()
    : kernelType_(KernelType::MATRIX_MULTIPLY)
    , relativeTolerance_(1e-5)
    , absoluteTolerance_(1e-8) {
}

GoldenReferenceValidator::~GoldenReferenceValidator() {
    cleanup();
}

void GoldenReferenceValidator::setup(const KernelConfig& config) {
    // Allocate reference data buffer based on expected output size
    size_t bufferSize = 0;
    
    switch (kernelType_) {
        case KernelType::MATRIX_MULTIPLY:
            bufferSize = config.matrixSize * config.matrixSize * sizeof(float);
            break;
        case KernelType::VECTOR_ADD:
            bufferSize = config.matrixSize * sizeof(float);
            break;
        case KernelType::REDUCTION:
            bufferSize = sizeof(float);  // Single value result
            break;
        case KernelType::CONVOLUTION:
            bufferSize = config.matrixSize * config.matrixSize * sizeof(float);
            break;
        case KernelType::CUSTOM:
            // Custom reference will handle its own allocation
            break;
    }
    
    if (bufferSize > 0) {
        referenceData_.resize(bufferSize);
    }
}

void GoldenReferenceValidator::cleanup() {
    referenceData_.clear();
}

ValidationResult GoldenReferenceValidator::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& config) {
    
    ValidationResult result;
    result.method = ValidationType::GOLDEN_REFERENCE;
    
    // Check if we need to convert non-CPU types to float32
    std::unique_ptr<float[]> convertedData;
    const void* dataToValidate = data;
    size_t effectiveElementSize = elementSize;
    
    if (ValidationTypeConverter::requiresConversion(elementSize)) {
        // Convert bf16/f16/fp8/fp4 to float32 for CPU validation
        try {
            convertedData = ValidationTypeConverter::convertToFloat32(data, numElements, elementSize);
            dataToValidate = convertedData.get();
            effectiveElementSize = sizeof(float);
            
            // Log conversion
            result.errorDetails = "Converted from non-CPU type to float32 for validation";
        } catch (const std::exception& e) {
            result.passed = false;
            result.errorDetails = std::string("Failed to convert data type: ") + e.what();
            result.confidence = 0.0;
            return result;
        }
    }
    
    // Compute reference based on kernel type
    if (kernelType_ == KernelType::CUSTOM && customReference_) {
        customReference_(referenceData_.data(), nullptr, numElements);
    } else {
        switch (kernelType_) {
            case KernelType::MATRIX_MULTIPLY:
                computeMatrixMultiplyReference(referenceData_.data(), config.matrixSize);
                break;
            case KernelType::VECTOR_ADD:
                computeVectorAddReference(referenceData_.data(), numElements);
                break;
            case KernelType::REDUCTION:
                computeReductionReference(referenceData_.data(), numElements);
                break;
            case KernelType::CONVOLUTION:
                computeConvolutionReference(referenceData_.data(), numElements);
                break;
            default:
                result.passed = false;
                result.errorDetails = "Unknown kernel type for golden reference";
                return result;
        }
    }
    
    // Compare results based on data type
    if (effectiveElementSize == sizeof(float) || effectiveElementSize == sizeof(double)) {
        result = compareFloatingPoint(dataToValidate, referenceData_.data(), numElements, effectiveElementSize);
    } else {
        result = compareInteger(dataToValidate, referenceData_.data(), numElements, effectiveElementSize);
    }
    
    // If we converted data, note it in the result
    if (convertedData) {
        if (result.passed) {
            result.errorDetails = "Validation passed (data converted from non-CPU type)";
        } else {
            result.errorDetails += " (data was converted from non-CPU type)";
        }
    }
    
    return result;
}

void GoldenReferenceValidator::computeMatrixMultiplyReference(void* output, size_t matrixSize) {
    float* C = static_cast<float*>(output);
    
    // Generate test matrices A and B
    std::vector<float> A(matrixSize * matrixSize);
    std::vector<float> B(matrixSize * matrixSize);
    
    // Initialize with deterministic pattern
    for (size_t i = 0; i < matrixSize * matrixSize; ++i) {
        A[i] = static_cast<float>(i % 100) / 100.0f;
        B[i] = static_cast<float>((i + 1) % 100) / 100.0f;
    }
    
    // Compute C = A * B
    GoldenReferenceHelpers::matrixMultiplyCPU(C, A.data(), B.data(), matrixSize);
}

void GoldenReferenceValidator::computeVectorAddReference(void* output, size_t numElements) {
    float* C = static_cast<float*>(output);
    
    // Generate test vectors
    std::vector<float> A(numElements);
    std::vector<float> B(numElements);
    
    for (size_t i = 0; i < numElements; ++i) {
        A[i] = static_cast<float>(i) / numElements;
        B[i] = static_cast<float>(numElements - i) / numElements;
    }
    
    GoldenReferenceHelpers::vectorAddCPU(C, A.data(), B.data(), numElements);
}

void GoldenReferenceValidator::computeReductionReference(void* output, size_t numElements) {
    float* result = static_cast<float*>(output);
    
    // Generate test data
    std::vector<float> data(numElements);
    for (size_t i = 0; i < numElements; ++i) {
        data[i] = static_cast<float>(i % 1000) / 1000.0f;
    }
    
    *result = GoldenReferenceHelpers::reductionCPU(data.data(), numElements);
}

void GoldenReferenceValidator::computeConvolutionReference(void* output, size_t numElements) {
    // Assuming square image for simplicity
    size_t width = static_cast<size_t>(std::sqrt(numElements));
    size_t height = width;
    
    float* outputData = static_cast<float*>(output);
    
    // Generate test input and kernel
    std::vector<float> input(numElements);
    std::vector<float> kernel(9);  // 3x3 kernel
    
    // Initialize input with pattern
    for (size_t i = 0; i < numElements; ++i) {
        input[i] = static_cast<float>(i % 256) / 256.0f;
    }
    
    // Simple edge detection kernel
    kernel[0] = -1; kernel[1] = -1; kernel[2] = -1;
    kernel[3] = -1; kernel[4] =  8; kernel[5] = -1;
    kernel[6] = -1; kernel[7] = -1; kernel[8] = -1;
    
    GoldenReferenceHelpers::convolution2DCPU(outputData, input.data(), kernel.data(), width, height, 3);
}

template<typename T>
ValidationResult GoldenReferenceValidator::compareData(const T* gpuData, const T* cpuData, size_t numElements) {
    ValidationResult result;
    result.method = ValidationType::GOLDEN_REFERENCE;
    result.passed = true;
    
    size_t mismatchCount = 0;
    double maxError = 0.0;
    size_t maxErrorIndex = 0;
    
    for (size_t i = 0; i < numElements; ++i) {
        T gpu = gpuData[i];
        T cpu = cpuData[i];
        
        bool match = false;
        double error = 0.0;
        
        if (std::is_floating_point<T>::value) {
            // Floating-point comparison with tolerance
            double absError = std::abs(static_cast<double>(gpu - cpu));
            double relError = absError / (std::abs(static_cast<double>(cpu)) + 1e-10);
            
            match = (absError <= absoluteTolerance_) || (relError <= relativeTolerance_);
            error = std::max(absError, relError);
        } else {
            // Exact comparison for integers
            match = (gpu == cpu);
            error = std::abs(static_cast<double>(gpu - cpu));
        }
        
        if (!match) {
            mismatchCount++;
            if (error > maxError) {
                maxError = error;
                maxErrorIndex = i;
            }
        }
    }
    
    if (mismatchCount > 0) {
        result.passed = false;
        result.corruptedElements = mismatchCount;
        result.confidence = 1.0 - (static_cast<double>(mismatchCount) / numElements);
        result.errorDetails = "Mismatch at " + std::to_string(mismatchCount) + " elements. "
                            + "Max error: " + std::to_string(maxError) + " at index " + std::to_string(maxErrorIndex);
    }
    
    return result;
}

ValidationResult GoldenReferenceValidator::compareFloatingPoint(
    const void* gpuData, const void* cpuData, size_t numElements, size_t elementSize) {
    
    if (elementSize == sizeof(float)) {
        return compareData(static_cast<const float*>(gpuData), 
                          static_cast<const float*>(cpuData), numElements);
    } else if (elementSize == sizeof(double)) {
        return compareData(static_cast<const double*>(gpuData),
                          static_cast<const double*>(cpuData), numElements);
    }
    
    ValidationResult result;
    result.passed = false;
    result.errorDetails = "Unsupported floating-point element size";
    return result;
}

ValidationResult GoldenReferenceValidator::compareInteger(
    const void* gpuData, const void* cpuData, size_t numElements, size_t elementSize) {
    
    ValidationResult result;
    result.method = ValidationType::GOLDEN_REFERENCE;
    result.passed = true;
    
    // Byte-wise comparison for integers
    size_t totalBytes = numElements * elementSize;
    const uint8_t* gpuBytes = static_cast<const uint8_t*>(gpuData);
    const uint8_t* cpuBytes = static_cast<const uint8_t*>(cpuData);
    
    size_t mismatchCount = 0;
    for (size_t i = 0; i < totalBytes; ++i) {
        if (gpuBytes[i] != cpuBytes[i]) {
            mismatchCount++;
        }
    }
    
    if (mismatchCount > 0) {
        result.passed = false;
        result.corruptedElements = mismatchCount / elementSize;  // Approximate element count
        result.confidence = 1.0 - (static_cast<double>(mismatchCount) / totalBytes);
        result.errorDetails = "Integer data mismatch: " + std::to_string(mismatchCount) + " bytes differ";
    }
    
    return result;
}

// Helper function implementations
namespace GoldenReferenceHelpers {
    
template<typename T>
void matrixMultiplyCPU(T* C, const T* A, const T* B, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

template<typename T>
void vectorAddCPU(T* C, const T* A, const T* B, size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

template<typename T>
T reductionCPU(const T* data, size_t N) {
    T sum = 0;
    for (size_t i = 0; i < N; ++i) {
        sum += data[i];
    }
    return sum;
}

template<typename T>
void convolution2DCPU(T* output, const T* input, const T* kernel,
                     size_t width, size_t height, size_t kernelSize) {
    int halfKernel = kernelSize / 2;
    
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            T sum = 0;
            
            for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    
                    // Handle boundaries with zero padding
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
                        sum += input[iy * width + ix] * kernel[kernelIdx];
                    }
                }
            }
            
            output[y * width + x] = sum;
        }
    }
}

// Explicit template instantiations
template void matrixMultiplyCPU<float>(float*, const float*, const float*, size_t);
template void matrixMultiplyCPU<double>(double*, const double*, const double*, size_t);
template void vectorAddCPU<float>(float*, const float*, const float*, size_t);
template void vectorAddCPU<double>(double*, const double*, const double*, size_t);
template float reductionCPU<float>(const float*, size_t);
template double reductionCPU<double>(const double*, size_t);
template void convolution2DCPU<float>(float*, const float*, const float*, size_t, size_t, size_t);
template void convolution2DCPU<double>(double*, const double*, const double*, size_t, size_t, size_t);

} // namespace GoldenReferenceHelpers