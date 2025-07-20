#include "mathematical_invariants.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cmath>
#include <algorithm>

// Matrix trace kernel
template<typename T>
__global__ void matrixTraceKernel(const T* matrix, size_t n, T* result) {
    extern __shared__ char sharedMem[];
    T* sharedData = reinterpret_cast<T*>(sharedMem);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    T localSum = 0;
    
    // Each thread handles multiple diagonal elements
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        localSum += matrix[i * n + i];
    }
    
    sharedData[tid] = localSum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

// Matrix norm kernel (Frobenius norm)
template<typename T>
__global__ void matrixNormKernel(const T* matrix, size_t rows, size_t cols, T* result, int normType) {
    extern __shared__ char sharedMem[];
    T* sharedData = reinterpret_cast<T*>(sharedMem);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t totalElements = rows * cols;
    
    T localSum = 0;
    
    if (normType == 2) {  // Frobenius norm
        for (size_t i = gid; i < totalElements; i += blockDim.x * gridDim.x) {
            T val = matrix[i];
            localSum += val * val;
        }
    } else if (normType == 1) {  // 1-norm (max column sum)
        // Simplified implementation
        for (size_t col = gid; col < cols; col += blockDim.x * gridDim.x) {
            T colSum = 0;
            for (size_t row = 0; row < rows; ++row) {
                colSum += abs(matrix[row * cols + col]);
            }
            localSum = max(localSum, colSum);
        }
    }
    
    sharedData[tid] = localSum;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (normType == 2) {
                sharedData[tid] += sharedData[tid + s];
            } else {
                sharedData[tid] = max(sharedData[tid], sharedData[tid + s]);
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        if (normType == 2) {
            atomicAdd(result, sharedData[0]);
        } else {
            T old = *result;
            while (old < sharedData[0]) {
                old = atomicCAS((unsigned int*)result, __float_as_uint(old), 
                               __float_as_uint(sharedData[0]));
            }
        }
    }
}

// Vector magnitude kernel
template<typename T>
__global__ void vectorMagnitudeKernel(const T* vector, size_t n, T* result) {
    extern __shared__ char sharedMem[];
    T* sharedData = reinterpret_cast<T*>(sharedMem);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    T localSum = 0;
    
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        T val = vector[i];
        localSum += val * val;
    }
    
    sharedData[tid] = localSum;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sharedData[0]);
    }
}

// Symmetry check kernel
template<typename T>
__global__ void symmetryCheckKernel(const T* matrix, size_t n, bool* isSymmetric) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n && row < col) {
        T a = matrix[row * n + col];
        T b = matrix[col * n + row];
        
        if (abs(a - b) > 1e-6) {
            *isSymmetric = false;
        }
    }
}

// Orthogonality check kernel (dot product)
template<typename T>
__global__ void orthogonalityCheckKernel(const T* v1, const T* v2, size_t n, T* dotProduct) {
    extern __shared__ char sharedMem[];
    T* sharedData = reinterpret_cast<T*>(sharedMem);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    T localSum = 0;
    
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        localSum += v1[i] * v2[i];
    }
    
    sharedData[tid] = localSum;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(dotProduct, sharedData[0]);
    }
}

MathematicalInvariantsValidator::MathematicalInvariantsValidator()
    : rows_(0)
    , cols_(0)
    , d_workspace_(nullptr)
    , workspaceSize_(0) {
}

MathematicalInvariantsValidator::~MathematicalInvariantsValidator() {
    cleanup();
}

void MathematicalInvariantsValidator::setup(const KernelConfig& config) {
    // Set default matrix dimensions based on config
    if (rows_ == 0 || cols_ == 0) {
        rows_ = config.matrixSize;
        cols_ = config.matrixSize;
    }
    
    // Allocate workspace
    size_t requiredSize = rows_ * cols_ * sizeof(double);
    if (workspaceSize_ < requiredSize) {
        if (d_workspace_) {
            cudaFree(d_workspace_);
        }
        cudaMalloc(&d_workspace_, requiredSize);
        workspaceSize_ = requiredSize;
    }
}

void MathematicalInvariantsValidator::cleanup() {
    if (d_workspace_) {
        cudaFree(d_workspace_);
        d_workspace_ = nullptr;
        workspaceSize_ = 0;
    }
    invariants_.clear();
}

void MathematicalInvariantsValidator::addInvariant(InvariantType type, double expectedValue, double tolerance) {
    InvariantCheck check;
    check.type = type;
    check.expectedValue = expectedValue;
    check.tolerance = tolerance;
    
    switch (type) {
        case InvariantType::MATRIX_TRACE:
            check.name = "Matrix Trace";
            break;
        case InvariantType::MATRIX_DETERMINANT:
            check.name = "Matrix Determinant";
            break;
        case InvariantType::MATRIX_RANK:
            check.name = "Matrix Rank";
            break;
        case InvariantType::MATRIX_NORM:
            check.name = "Matrix Norm";
            break;
        case InvariantType::VECTOR_MAGNITUDE:
            check.name = "Vector Magnitude";
            break;
        case InvariantType::ORTHOGONALITY:
            check.name = "Orthogonality";
            break;
        case InvariantType::SYMMETRY:
            check.name = "Symmetry";
            break;
        case InvariantType::POSITIVE_DEFINITENESS:
            check.name = "Positive Definiteness";
            break;
        case InvariantType::CONSERVATION_LAW:
            check.name = "Conservation Law";
            break;
        default:
            check.name = "Unknown Invariant";
    }
    
    invariants_.push_back(check);
}

void MathematicalInvariantsValidator::addCustomInvariant(
    const std::string& name, InvariantFunction func, double expectedValue, double tolerance) {
    
    InvariantCheck check;
    check.type = InvariantType::CUSTOM;
    check.name = name;
    check.customFunc = func;
    check.expectedValue = expectedValue;
    check.tolerance = tolerance;
    
    invariants_.push_back(check);
}

ValidationResult MathematicalInvariantsValidator::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& config) {
    
    ValidationResult result;
    result.method = ValidationType::MATHEMATICAL_INVARIANT;
    result.passed = true;
    
    for (const auto& check : invariants_) {
        double computedValue = 0.0;
        
        if (check.type == InvariantType::CUSTOM && check.customFunc) {
            computedValue = check.customFunc(data, numElements);
        } else {
            switch (check.type) {
                case InvariantType::MATRIX_TRACE:
                    computedValue = computeMatrixTrace(data, numElements, elementSize);
                    break;
                case InvariantType::MATRIX_DETERMINANT:
                    computedValue = computeMatrixDeterminant(data, numElements, elementSize);
                    break;
                case InvariantType::MATRIX_RANK:
                    computedValue = computeMatrixRank(data, numElements, elementSize);
                    break;
                case InvariantType::MATRIX_NORM:
                    computedValue = computeMatrixNorm(data, numElements, elementSize, 2);  // Frobenius norm
                    break;
                case InvariantType::VECTOR_MAGNITUDE:
                    computedValue = computeVectorMagnitude(data, numElements, elementSize);
                    break;
                case InvariantType::ORTHOGONALITY:
                    computedValue = checkOrthogonality(data, numElements, elementSize);
                    break;
                case InvariantType::SYMMETRY:
                    computedValue = checkSymmetry(data, numElements, elementSize);
                    break;
                case InvariantType::POSITIVE_DEFINITENESS:
                    computedValue = checkPositiveDefiniteness(data, numElements, elementSize);
                    break;
                case InvariantType::CONSERVATION_LAW:
                    computedValue = checkConservationLaw(data, numElements, elementSize);
                    break;
                default:
                    continue;
            }
        }
        
        if (!validateInvariant(computedValue, check.expectedValue, check.tolerance)) {
            result.passed = false;
            result.corruptedElements++;
            if (result.errorDetails.empty()) {
                result.errorDetails = "Failed invariants: ";
            }
            result.errorDetails += check.name + " (expected: " + std::to_string(check.expectedValue) +
                                 ", got: " + std::to_string(computedValue) + "); ";
        }
    }
    
    if (!result.passed) {
        result.confidence = 1.0 - (static_cast<double>(result.corruptedElements) / invariants_.size());
    }
    
    return result;
}

double MathematicalInvariantsValidator::computeMatrixTrace(
    const void* data, size_t numElements, size_t elementSize) {
    
    if (elementSize == sizeof(float)) {
        const float* matrix = static_cast<const float*>(data);
        float h_result = 0.0f;
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (rows_ + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(float);
        
        matrixTraceKernel<<<gridSize, blockSize, sharedMemSize>>>(matrix, rows_, d_result);
        
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        return static_cast<double>(h_result);
    } else if (elementSize == sizeof(double)) {
        const double* matrix = static_cast<const double*>(data);
        double h_result = 0.0;
        double* d_result;
        cudaMalloc(&d_result, sizeof(double));
        cudaMemcpy(d_result, &h_result, sizeof(double), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (rows_ + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(double);
        
        matrixTraceKernel<<<gridSize, blockSize, sharedMemSize>>>(matrix, rows_, d_result);
        
        cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        return h_result;
    }
    
    return 0.0;
}

double MathematicalInvariantsValidator::computeMatrixDeterminant(
    const void* data, size_t numElements, size_t elementSize) {
    
    // Use cuBLAS/cuSOLVER for determinant computation
    // Simplified implementation - computing determinant via LU decomposition
    
    cusolverDnHandle_t cusolverH = nullptr;
    cusolverDnCreate(&cusolverH);
    
    double det = 0.0;
    
    if (elementSize == sizeof(float)) {
        // Implementation for float matrices
        const float* matrix = static_cast<const float*>(data);
        
        // Copy matrix to workspace
        float* d_A = static_cast<float*>(d_workspace_);
        cudaMemcpy(d_A, matrix, rows_ * cols_ * sizeof(float), cudaMemcpyDeviceToDevice);
        
        // Compute determinant using LU decomposition
        // (Simplified - actual implementation would use cuSOLVER)
        det = 1.0;  // Placeholder
        
    } else if (elementSize == sizeof(double)) {
        // Implementation for double matrices
        // const double* matrix = static_cast<const double*>(data);  // TODO: Will be used when implementing actual determinant calculation
        
        // Similar implementation for double
        det = 1.0;  // Placeholder
    }
    
    cusolverDnDestroy(cusolverH);
    return det;
}

double MathematicalInvariantsValidator::computeMatrixRank(
    const void* data, size_t numElements, size_t elementSize) {
    
    // Compute matrix rank using SVD
    // Simplified implementation
    return static_cast<double>(std::min(rows_, cols_));  // Placeholder
}

double MathematicalInvariantsValidator::computeMatrixNorm(
    const void* data, size_t numElements, size_t elementSize, int normType) {
    
    if (elementSize == sizeof(float)) {
        const float* matrix = static_cast<const float*>(data);
        float h_result = 0.0f;
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (rows_ * cols_ + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(float);
        
        matrixNormKernel<<<gridSize, blockSize, sharedMemSize>>>(
            matrix, rows_, cols_, d_result, normType);
        
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        if (normType == 2) {  // Frobenius norm
            return std::sqrt(static_cast<double>(h_result));
        }
        return static_cast<double>(h_result);
    }
    
    return 0.0;
}

double MathematicalInvariantsValidator::computeVectorMagnitude(
    const void* data, size_t numElements, size_t elementSize) {
    
    if (elementSize == sizeof(float)) {
        const float* vector = static_cast<const float*>(data);
        float h_result = 0.0f;
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (numElements + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(float);
        
        vectorMagnitudeKernel<<<gridSize, blockSize, sharedMemSize>>>(
            vector, numElements, d_result);
        
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        return std::sqrt(static_cast<double>(h_result));
    }
    
    return 0.0;
}

double MathematicalInvariantsValidator::checkOrthogonality(
    const void* data, size_t numElements, size_t elementSize) {
    
    // Check if vectors are orthogonal (dot product should be ~0)
    // Simplified: assumes two vectors concatenated
    size_t vecSize = numElements / 2;
    
    if (elementSize == sizeof(float)) {
        const float* v1 = static_cast<const float*>(data);
        const float* v2 = v1 + vecSize;
        
        float h_result = 0.0f;
        float* d_result;
        cudaMalloc(&d_result, sizeof(float));
        cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int gridSize = (vecSize + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(float);
        
        orthogonalityCheckKernel<<<gridSize, blockSize, sharedMemSize>>>(
            v1, v2, vecSize, d_result);
        
        cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        
        return static_cast<double>(h_result);
    }
    
    return 0.0;
}

double MathematicalInvariantsValidator::checkSymmetry(
    const void* data, size_t numElements, size_t elementSize) {
    
    if (elementSize == sizeof(float)) {
        const float* matrix = static_cast<const float*>(data);
        
        bool h_isSymmetric = true;
        bool* d_isSymmetric;
        cudaMalloc(&d_isSymmetric, sizeof(bool));
        cudaMemcpy(d_isSymmetric, &h_isSymmetric, sizeof(bool), cudaMemcpyHostToDevice);
        
        dim3 blockDim(16, 16);
        dim3 gridDim((rows_ + blockDim.x - 1) / blockDim.x,
                     (cols_ + blockDim.y - 1) / blockDim.y);
        
        symmetryCheckKernel<<<gridDim, blockDim>>>(matrix, rows_, d_isSymmetric);
        
        cudaMemcpy(&h_isSymmetric, d_isSymmetric, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(d_isSymmetric);
        
        return h_isSymmetric ? 1.0 : 0.0;
    }
    
    return 0.0;
}

double MathematicalInvariantsValidator::checkPositiveDefiniteness(
    const void* data, size_t numElements, size_t elementSize) {
    
    // Check if all eigenvalues are positive
    // Simplified implementation
    return 1.0;  // Placeholder
}

double MathematicalInvariantsValidator::checkConservationLaw(
    const void* data, size_t numElements, size_t elementSize) {
    
    // Check conservation properties (e.g., sum of elements)
    // This is a simplified version that computes the sum
    
    if (elementSize == sizeof(float)) {
        const float* values = static_cast<const float*>(data);
        
        // Use thrust or cub for reduction
        float sum = 0.0f;
        std::vector<float> h_data(numElements);
        cudaMemcpy(h_data.data(), values, numElements * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < numElements; ++i) {
            sum += h_data[i];
        }
        
        return static_cast<double>(sum);
    }
    
    return 0.0;
}

bool MathematicalInvariantsValidator::validateInvariant(
    double computed, double expected, double tolerance) {
    
    double absError = std::abs(computed - expected);
    double relError = absError / (std::abs(expected) + 1e-10);
    
    return (absError <= tolerance) || (relError <= tolerance);
}

// Explicit template instantiations
template __global__ void matrixTraceKernel<float>(const float*, size_t, float*);
template __global__ void matrixTraceKernel<double>(const double*, size_t, double*);
template __global__ void matrixNormKernel<float>(const float*, size_t, size_t, float*, int);
template __global__ void matrixNormKernel<double>(const double*, size_t, size_t, double*, int);
template __global__ void vectorMagnitudeKernel<float>(const float*, size_t, float*);
template __global__ void vectorMagnitudeKernel<double>(const double*, size_t, double*);
template __global__ void symmetryCheckKernel<float>(const float*, size_t, bool*);
template __global__ void symmetryCheckKernel<double>(const double*, size_t, bool*);
template __global__ void orthogonalityCheckKernel<float>(const float*, const float*, size_t, float*);
template __global__ void orthogonalityCheckKernel<double>(const double*, const double*, size_t, double*);