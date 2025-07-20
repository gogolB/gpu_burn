#ifndef VALIDATION_TYPE_CONVERTER_H
#define VALIDATION_TYPE_CONVERTER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "gpu_types.h"
#include <vector>
#include <memory>

// Type converter for validation purposes
class ValidationTypeConverter {
public:
    // Convert various GPU types to float32 for CPU validation
    static std::unique_ptr<float[]> convertToFloat32(
        const void* data,
        size_t numElements,
        size_t elementSize);
    
    // Type detection helpers
    static bool isBF16Type(size_t elementSize);
    static bool isFP16Type(size_t elementSize);
    static bool isFP8Type(size_t elementSize);
    static bool isBF8Type(size_t elementSize);
    static bool isFP4Type(size_t elementSize);
    static bool requiresConversion(size_t elementSize);
    
    // GPU kernels for type conversion
    static void convertBF16ToFloat32GPU(const __nv_bfloat16* input, float* output, size_t numElements);
    static void convertFP16ToFloat32GPU(const __half* input, float* output, size_t numElements);
    static void convertFP8ToFloat32GPU(const fp8_t* input, float* output, size_t numElements);
    static void convertBF8ToFloat32GPU(const bf8_t* input, float* output, size_t numElements);
    static void convertFP4ToFloat32GPU(const fp4_t* input, float* output, size_t numElements);
    
    // Alternative: Skip validation for unsupported types
    static bool shouldSkipValidation(size_t elementSize);
};

// Conversion kernels
__global__ void bf16ToFloat32Kernel(const __nv_bfloat16* input, float* output, size_t numElements);
__global__ void fp16ToFloat32Kernel(const __half* input, float* output, size_t numElements);
__global__ void fp8ToFloat32Kernel(const fp8_t* input, float* output, size_t numElements);
__global__ void bf8ToFloat32Kernel(const bf8_t* input, float* output, size_t numElements);
__global__ void fp4ToFloat32Kernel(const fp4_t* input, float* output, size_t numElements);

#endif // VALIDATION_TYPE_CONVERTER_H