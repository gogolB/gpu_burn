#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>

// BF16 Matrix multiplication kernel using shared memory
template<int TILE_SIZE>
__global__ void matmul_bf16_kernel(const __nv_bfloat16* __restrict__ A, 
                                   const __nv_bfloat16* __restrict__ B, 
                                   __nv_bfloat16* __restrict__ C, 
                                   int N) {
    __shared__ __nv_bfloat16 As[TILE_SIZE][TILE_SIZE];
    __shared__ __nv_bfloat16 Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;  // Use float for accumulation
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = __float2bfloat16(0.0f);
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = __float2bfloat16(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += __bfloat162float(As[ty][k]) * __bfloat162float(Bs[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = __float2bfloat16(sum);
    }
}

// High-intensity BF16 kernel that stresses the GPU
__global__ void stress_bf16_kernel(__nv_bfloat16* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = __bfloat162float(data[idx]);
        
        // Perform intensive computations
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
            val = val / 1.0001f;
            val = sqrtf(fabsf(val));
            val = val * val + 0.5f;
            val = fmaf(val, 1.001f, 0.001f);
            val = expf(val * 0.001f);
            val = logf(fabsf(val) + 1.0f);
            val = sinf(val * 0.1f);
            val = cosf(val * 0.1f);
            val = tanhf(val * 0.1f);
        }
        
        data[idx] = __float2bfloat16(val);
    }
}

// BF16 kernel implementation
class BF16Kernel : public TypedKernel<__nv_bfloat16> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    BF16Kernel() {
        cublasCreate(&cublasHandle_);
        // Enable tensor cores for BF16
        cublasSetMathMode(cublasHandle_, CUBLAS_TENSOR_OP_MATH);
    }
    
    ~BF16Kernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Check BF16 support
            GpuInfo info = getGpuInfo(config.deviceId);
            if (!info.supportsBF16()) {
                result.success = false;
                result.errorMessage = "BF16 not supported on this GPU (requires Ampere or newer)";
                return result;
            }
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<__nv_bfloat16> d_A(elements);
            DeviceBuffer<__nv_bfloat16> d_B(elements);
            DeviceBuffer<__nv_bfloat16> d_C(elements);
            
            // Initialize matrices
            __nv_bfloat16* h_A = new __nv_bfloat16[elements];
            __nv_bfloat16* h_B = new __nv_bfloat16[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = __float2bfloat16(1.0f + (i % 100) * 0.01f);
                h_B[i] = __float2bfloat16(2.0f - (i % 100) * 0.01f);
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
                    // Use cuBLAS for tensor core operations on BF16
                    float alpha = 1.0f, beta = 0.0f;
                    cublasGemmEx(cublasHandle_, 
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                d_B.get(), CUDA_R_16BF, N,
                                d_A.get(), CUDA_R_16BF, N,
                                &beta,
                                d_C.get(), CUDA_R_16BF, N,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                } else {
                    // Use custom kernel
                    matmul_bf16_kernel<TILE_SIZE><<<grid, block>>>(
                        d_A.get(), d_B.get(), d_C.get(), N);
                }
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_bf16_kernel<<<stressBlocks, 256>>>(d_C.get(), elements, 15);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(__nv_bfloat16) * config.numIterations;
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
        return "BF16 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity BF16 matrix multiplication with tensor core support (Ampere+)";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        return info.supportsBF16();
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        return 3 * config.matrixSize * config.matrixSize * sizeof(__nv_bfloat16);
    }
};

// Register kernel
REGISTER_KERNEL(BF16Kernel)