#include "validation_type_converter.h"
#include <cuda_runtime.h>
#include <cstring>

// GPU kernels for type conversion
__global__ void bf16ToFloat32Kernel(const __nv_bfloat16* input, float* output, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

__global__ void fp16ToFloat32Kernel(const __half* input, float* output, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        output[idx] = __half2float(input[idx]);
    }
}

__global__ void fp8ToFloat32Kernel(const fp8_t* input, float* output, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Convert FP8 to float using the cast operator
        output[idx] = static_cast<float>(input[idx]);
    }
}

__global__ void bf8ToFloat32Kernel(const bf8_t* input, float* output, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Convert BF8 to float using the cast operator
        output[idx] = static_cast<float>(input[idx]);
    }
}

__global__ void fp4ToFloat32Kernel(const fp4_t* input, float* output, size_t numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        // Convert FP4 to float using the cast operator
        output[idx] = static_cast<float>(input[idx]);
    }
}

// Static methods implementation
std::unique_ptr<float[]> ValidationTypeConverter::convertToFloat32(
    const void* data, 
    size_t numElements, 
    size_t elementSize) {
    
    // Allocate output buffer
    std::unique_ptr<float[]> floatData(new float[numElements]);
    
    // Allocate device memory for conversion
    float* d_output = nullptr;
    cudaMalloc(&d_output, numElements * sizeof(float));
    
    // Launch appropriate conversion kernel
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    
    if (isBF16Type(elementSize)) {
        bf16ToFloat32Kernel<<<gridSize, blockSize>>>(
            static_cast<const __nv_bfloat16*>(data), d_output, numElements);
    } else if (isFP16Type(elementSize)) {
        fp16ToFloat32Kernel<<<gridSize, blockSize>>>(
            static_cast<const __half*>(data), d_output, numElements);
    } else if (isFP8Type(elementSize)) {
        fp8ToFloat32Kernel<<<gridSize, blockSize>>>(
            static_cast<const fp8_t*>(data), d_output, numElements);
    } else if (isBF8Type(elementSize)) {
        bf8ToFloat32Kernel<<<gridSize, blockSize>>>(
            static_cast<const bf8_t*>(data), d_output, numElements);
    } else if (isFP4Type(elementSize)) {
        // FP4 is packed - need special handling
        // For now, treat as individual elements
        fp4ToFloat32Kernel<<<gridSize, blockSize>>>(
            static_cast<const fp4_t*>(data), d_output, numElements);
    } else if (elementSize == sizeof(float)) {
        // Already float, just copy
        cudaMemcpy(d_output, data, numElements * sizeof(float), cudaMemcpyDeviceToDevice);
    } else if (elementSize == sizeof(double)) {
        // For double, we need a different conversion
        // For simplicity, we'll do it on CPU
        double* hostDouble = new double[numElements];
        cudaMemcpy(hostDouble, data, numElements * sizeof(double), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < numElements; i++) {
            floatData[i] = static_cast<float>(hostDouble[i]);
        }
        delete[] hostDouble;
        cudaFree(d_output);
        return floatData;
    }
    
    // Copy result back to host
    cudaMemcpy(floatData.get(), d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_output);
    
    return floatData;
}

bool ValidationTypeConverter::isBF16Type(size_t elementSize) {
    return elementSize == sizeof(__nv_bfloat16);
}

bool ValidationTypeConverter::isFP16Type(size_t elementSize) {
    return elementSize == sizeof(__half);
}

bool ValidationTypeConverter::isFP8Type(size_t elementSize) {
    return elementSize == sizeof(fp8_t);
}

bool ValidationTypeConverter::isBF8Type(size_t elementSize) {
    return elementSize == sizeof(bf8_t);
}

bool ValidationTypeConverter::isFP4Type(size_t elementSize) {
    // FP4 is tricky as it's 4 bits, not a full byte
    // We'll assume if elementSize is 1 and not FP8/BF8, it might be FP4
    return false; // For now, handle this specially
}

bool ValidationTypeConverter::requiresConversion(size_t elementSize) {
    return isBF16Type(elementSize) || 
           isFP16Type(elementSize) || 
           isFP8Type(elementSize) || 
           isBF8Type(elementSize) ||
           isFP4Type(elementSize);
}

bool ValidationTypeConverter::shouldSkipValidation(size_t elementSize) {
    // For extremely experimental types, we might want to skip validation
    // For now, we'll try to convert all types
    return false;
}

void ValidationTypeConverter::convertBF16ToFloat32GPU(const __nv_bfloat16* input, float* output, size_t numElements) {
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    bf16ToFloat32Kernel<<<gridSize, blockSize>>>(input, output, numElements);
    cudaDeviceSynchronize();
}

void ValidationTypeConverter::convertFP16ToFloat32GPU(const __half* input, float* output, size_t numElements) {
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    fp16ToFloat32Kernel<<<gridSize, blockSize>>>(input, output, numElements);
    cudaDeviceSynchronize();
}

void ValidationTypeConverter::convertFP8ToFloat32GPU(const fp8_t* input, float* output, size_t numElements) {
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    fp8ToFloat32Kernel<<<gridSize, blockSize>>>(input, output, numElements);
    cudaDeviceSynchronize();
}

void ValidationTypeConverter::convertBF8ToFloat32GPU(const bf8_t* input, float* output, size_t numElements) {
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    bf8ToFloat32Kernel<<<gridSize, blockSize>>>(input, output, numElements);
    cudaDeviceSynchronize();
}

void ValidationTypeConverter::convertFP4ToFloat32GPU(const fp4_t* input, float* output, size_t numElements) {
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;
    fp4ToFloat32Kernel<<<gridSize, blockSize>>>(input, output, numElements);
    cudaDeviceSynchronize();
}