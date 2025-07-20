#include "kernel_interface.h"
#include "gpu_utils.h"
#include "gpu_types.h"

// FP4 Matrix multiplication kernel
// Note: This is an experimental implementation for 4-bit floating point
template<int TILE_SIZE>
__global__ void matmul_fp4_kernel(const fp4_t* __restrict__ A, 
                                  const fp4_t* __restrict__ B, 
                                  fp4_t* __restrict__ C, 
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
    
    // Write result, converting back to FP4
    if (row < N && col < N) {
        C[row * N + col] = fp4_t(sum);
    }
}

// High-intensity FP4 kernel that stresses the GPU
__global__ void stress_fp4_kernel(fp4_t* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float val = static_cast<float>(data[idx]);
        
        // Perform intensive computations in float precision
        for (int i = 0; i < iterations; i++) {
            val = val * 1.01f + 0.01f;
            val = val / 1.01f;
            val = sqrtf(fabsf(val));
            val = val * val + 0.1f;
            // Keep values in FP4 representable range
            val = fminf(fmaxf(val, 0.0f), 7.0f);
        }
        
        data[idx] = fp4_t(val);
    }
}

// FP4 kernel implementation
class FP4Kernel : public TypedKernel<fp4_t> {
public:
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Check GPU architecture - FP4 is experimental
            GpuInfo info = getGpuInfo(config.deviceId);
            if (info.computeCapabilityMajor < 9) {
                result.success = false;
                result.errorMessage = "FP4 is experimental and requires Hopper or newer architecture";
                return result;
            }
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<fp4_t> d_A(elements);
            DeviceBuffer<fp4_t> d_B(elements);
            DeviceBuffer<fp4_t> d_C(elements);
            
            // Initialize matrices
            fp4_t* h_A = new fp4_t[elements];
            fp4_t* h_B = new fp4_t[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = fp4_t(1.0f + (i % 4) * 0.25f);
                h_B[i] = fp4_t(2.0f - (i % 4) * 0.25f);
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
                matmul_fp4_kernel<TILE_SIZE><<<grid, block>>>(
                    d_A.get(), d_B.get(), d_C.get(), N);
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_fp4_kernel<<<stressBlocks, 256>>>(d_C.get(), elements, 40);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(fp4_t) * config.numIterations;
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
        return "FP4 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "Experimental 4-bit floating point matrix multiplication (Hopper+)";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        // FP4 is experimental, requiring at least Hopper
        return info.computeCapabilityMajor >= 9;
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        // FP4 uses 4 bits per element, but we allocate as uint8_t
        return 3 * config.matrixSize * config.matrixSize * sizeof(uint8_t) / 2;
    }
};

// Register kernel
REGISTER_KERNEL(FP4Kernel)