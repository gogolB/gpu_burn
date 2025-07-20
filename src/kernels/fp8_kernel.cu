#include "kernel_interface.h"
#include "gpu_utils.h"
#include "gpu_types.h"
#include <cublas_v2.h>

// FP8 Matrix multiplication kernel
// Note: This is a simplified implementation as FP8 support is hardware-specific
template<int TILE_SIZE>
__global__ void matmul_fp8_kernel(const fp8_t* __restrict__ A, 
                                  const fp8_t* __restrict__ B, 
                                  fp8_t* __restrict__ C, 
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
        // Load tile into shared memory, converting to float
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = static_cast<float>(A[row * N + t * TILE_SIZE + tx]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = static_cast<float>(B[(t * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product in float
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result, converting back to FP8
    if (row < N && col < N) {
        C[row * N + col] = fp8_t(sum);
    }
}

// High-intensity FP8 kernel that stresses the GPU
__global__ void stress_fp8_kernel(fp8_t* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = static_cast<float>(data[idx]);
        
        // Perform intensive computations in float precision
        for (int i = 0; i < iterations; i++) {
            val = val * 1.001f + 0.001f;
            val = val / 1.001f;
            val = sqrtf(fabsf(val));
            val = val * val + 0.5f;
            val = fmaf(val, 1.001f, 0.001f);
            // Keep values in FP8 range
            val = fminf(fmaxf(val, -15.0f), 15.0f);
        }
        
        data[idx] = fp8_t(val);
    }
}

// FP8 kernel implementation
class FP8Kernel : public TypedKernel<fp8_t> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    FP8Kernel() {
        cublasCreate(&cublasHandle_);
    }
    
    ~FP8Kernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Check FP8 support
            GpuInfo info = getGpuInfo(config.deviceId);
            if (!info.supportsFP8()) {
                result.success = false;
                result.errorMessage = "FP8 not supported on this GPU (requires Ada Lovelace or newer)";
                return result;
            }
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<fp8_t> d_A(elements);
            DeviceBuffer<fp8_t> d_B(elements);
            DeviceBuffer<fp8_t> d_C(elements);
            
            // Initialize matrices
            fp8_t* h_A = new fp8_t[elements];
            fp8_t* h_B = new fp8_t[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = fp8_t(1.0f + (i % 10) * 0.1f);
                h_B[i] = fp8_t(2.0f - (i % 10) * 0.1f);
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
                // FP8 doesn't have direct cuBLAS support yet, use custom kernel
                matmul_fp8_kernel<TILE_SIZE><<<grid, block>>>(
                    d_A.get(), d_B.get(), d_C.get(), N);
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_fp8_kernel<<<stressBlocks, 256>>>(d_C.get(), elements, 25);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(fp8_t) * config.numIterations;
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
        return "FP8 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity FP8 matrix multiplication (Ada Lovelace+)";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        return info.supportsFP8();
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        return 3 * config.matrixSize * config.matrixSize * sizeof(fp8_t);
    }
};

// Register kernel
REGISTER_KERNEL(FP8Kernel)