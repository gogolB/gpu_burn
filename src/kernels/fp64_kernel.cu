#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>

// FP64 Matrix multiplication kernel using shared memory
template<int TILE_SIZE>
__global__ void matmul_fp64_kernel(const double* __restrict__ A, 
                                   const double* __restrict__ B, 
                                   double* __restrict__ C, 
                                   int N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
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

// High-intensity FP64 kernel that stresses the GPU
__global__ void stress_fp64_kernel(double* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        double val = data[idx];
        
        // Perform intensive computations
        for (int i = 0; i < iterations; i++) {
            // Mix of operations to stress different units
            val = val * 1.0001 + 0.0001;
            val = val / 1.0001;
            val = sqrt(fabs(val));
            val = val * val + 0.5;
            val = fma(val, 1.001, 0.001);
            val = exp(val * 0.001);
            val = log(fabs(val) + 1.0);
            val = sin(val);
            val = cos(val);
            val = tan(val * 0.1);
            val = atan(val);
            val = asin(fmod(val, 1.0));
            val = acos(fmod(val, 1.0));
        }
        
        data[idx] = val;
    }
}

// FP64 kernel implementation
class FP64Kernel : public TypedKernel<double> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    FP64Kernel() {
        cublasCreate(&cublasHandle_);
    }
    
    ~FP64Kernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<double> d_A(elements);
            DeviceBuffer<double> d_B(elements);
            DeviceBuffer<double> d_C(elements);
            
            // Initialize matrices
            double* h_A = new double[elements];
            double* h_B = new double[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = 1.0 + (i % 100) * 0.01;
                h_B[i] = 2.0 - (i % 100) * 0.01;
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
                    // Use cuBLAS (no tensor cores for FP64)
                    double alpha = 1.0, beta = 0.0;
                    cublasDgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N, N, &alpha,
                               d_B.get(), N,
                               d_A.get(), N,
                               &beta,
                               d_C.get(), N);
                } else {
                    // Use custom kernel
                    matmul_fp64_kernel<TILE_SIZE><<<grid, block>>>(
                        d_A.get(), d_B.get(), d_C.get(), N);
                }
                
                // Also run stress kernel
                int stressBlocks = (elements + 127) / 128;  // Fewer threads for FP64
                stress_fp64_kernel<<<stressBlocks, 128>>>(d_C.get(), elements, 8);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(double) * config.numIterations;
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
        return "FP64 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity FP64 matrix multiplication and stress test";
    }
    
    bool isSupported(int deviceId) const override {
        // FP64 is supported on all CUDA devices, but performance varies
        GpuInfo info = getGpuInfo(deviceId);
        return info.computeCapabilityMajor >= 2;  // Fermi and newer
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        return 3 * config.matrixSize * config.matrixSize * sizeof(double);
    }
};

// Register kernel
REGISTER_KERNEL(FP64Kernel)