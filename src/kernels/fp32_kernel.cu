#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>

// FP32 Matrix multiplication kernel using shared memory
template<int TILE_SIZE>
__global__ void matmul_fp32_kernel(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C, 
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// High-intensity FP32 kernel that stresses the GPU
__global__ void stress_fp32_kernel(float* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = data[idx];
        
        // Perform intensive computations
        for (int i = 0; i < iterations; i++) {
            // Mix of operations to stress different units
            val = val * 1.0001f + 0.0001f;
            val = __fdividef(val, 1.0001f);
            val = sqrtf(fabsf(val));
            val = val * val + 0.5f;
            val = fmaf(val, 1.001f, 0.001f);
            val = __expf(val * 0.001f);
            val = __logf(fabsf(val) + 1.0f);
            val = sinf(val);
            val = cosf(val);
            val = tanf(val * 0.1f);
        }
        
        data[idx] = val;
    }
}

// FP32 kernel implementation
class FP32Kernel : public TypedKernel<float> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    FP32Kernel() {
        cublasCreate(&cublasHandle_);
    }
    
    ~FP32Kernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<float> d_A(elements);
            DeviceBuffer<float> d_B(elements);
            DeviceBuffer<float> d_C(elements);
            
            // Initialize matrices
            float* h_A = new float[elements];
            float* h_B = new float[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = 1.0f + (i % 100) * 0.01f;
                h_B[i] = 2.0f - (i % 100) * 0.01f;
            }
            
            copyToDevice(d_A.get(), h_A, elements);
            copyToDevice(d_B.get(), h_B, elements);
            
            // Configure kernel launch
            const int TILE_SIZE = 16;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
            
            // Start timing
            CudaTimer timer;
            timer.start();
            
            // Run iterations
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                if (config.useTensorCores) {
                    // Use cuBLAS for tensor core operations
                    float alpha = 1.0f, beta = 0.0f;
                    cublasSgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N, N, &alpha,
                               d_B.get(), N,
                               d_A.get(), N,
                               &beta,
                               d_C.get(), N);
                } else {
                    // Use custom kernel
                    matmul_fp32_kernel<TILE_SIZE><<<grid, block>>>(
                        d_A.get(), d_B.get(), d_C.get(), N);
                }
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_fp32_kernel<<<stressBlocks, 256>>>(d_C.get(), elements, 10);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(float) * config.numIterations;
            result.memoryBandwidthGBps = (totalBytes / 1e9) / (result.executionTimeMs / 1000.0);
            
            result.success = true;
            
            delete[] h_A;
            delete[] h_B;
            
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = e.what();
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "FP32 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity FP32 matrix multiplication and stress test";
    }
    
    bool isSupported(int deviceId) const override {
        // FP32 is supported on all CUDA devices
        return true;
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        return 3 * config.matrixSize * config.matrixSize * sizeof(float);
    }
};

// Register kernel
REGISTER_KERNEL(FP32Kernel)