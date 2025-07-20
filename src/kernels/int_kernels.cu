#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>

// UINT8 Matrix multiplication kernel
template<int TILE_SIZE>
__global__ void matmul_uint8_kernel(const uint8_t* __restrict__ A, 
                                    const uint8_t* __restrict__ B, 
                                    uint8_t* __restrict__ C, 
                                    int N) {
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result (with saturation)
    if (row < N && col < N) {
        C[row * N + col] = (sum > 255) ? 255 : (uint8_t)sum;
    }
}

// UINT16 Matrix multiplication kernel
template<int TILE_SIZE>
__global__ void matmul_uint16_kernel(const uint16_t* __restrict__ A, 
                                     const uint16_t* __restrict__ B, 
                                     uint16_t* __restrict__ C, 
                                     int N) {
    __shared__ int As[TILE_SIZE][TILE_SIZE];
    __shared__ int Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    int sum = 0;
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0;
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result (with saturation)
    if (row < N && col < N) {
        C[row * N + col] = (sum > 65535) ? 65535 : (uint16_t)sum;
    }
}

// High-intensity integer stress kernel
template<typename T>
__global__ void stress_int_kernel(T* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        T val = data[idx];
        
        // Perform intensive integer operations
        for (int i = 0; i < iterations; i++) {
            val = val * 3 + 1;
            val = val >> 1;
            val = val ^ (val << 2);
            val = val | (val >> 3);
            val = val & 0xFF;
            val = val + (val >> 4);
            val = __popc(val) + val;
        }
        
        data[idx] = val;
    }
}

// UINT8 kernel implementation
class UINT8Kernel : public TypedKernel<uint8_t> {
public:
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<uint8_t> d_A(elements);
            DeviceBuffer<uint8_t> d_B(elements);
            DeviceBuffer<uint8_t> d_C(elements);
            
            // Initialize matrices
            uint8_t* h_A = new uint8_t[elements];
            uint8_t* h_B = new uint8_t[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = (i % 256);
                h_B[i] = ((i + 128) % 256);
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
                matmul_uint8_kernel<TILE_SIZE><<<grid, block>>>(
                    d_A.get(), d_B.get(), d_C.get(), N);
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_int_kernel<uint8_t><<<stressBlocks, 256>>>(
                    d_C.get(), elements, 30);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(uint8_t) * config.numIterations;
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
        return "UINT8 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity UINT8 matrix multiplication and integer operations";
    }
    
    bool isSupported(int deviceId) const override {
        return true; // Integer operations are supported on all GPUs
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        return 3 * config.matrixSize * config.matrixSize * sizeof(uint8_t);
    }
};

// UINT16 kernel implementation
class UINT16Kernel : public TypedKernel<uint16_t> {
public:
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<uint16_t> d_A(elements);
            DeviceBuffer<uint16_t> d_B(elements);
            DeviceBuffer<uint16_t> d_C(elements);
            
            // Initialize matrices
            uint16_t* h_A = new uint16_t[elements];
            uint16_t* h_B = new uint16_t[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = (i % 65536);
                h_B[i] = ((i + 32768) % 65536);
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
                matmul_uint16_kernel<TILE_SIZE><<<grid, block>>>(
                    d_A.get(), d_B.get(), d_C.get(), N);
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_int_kernel<uint16_t><<<stressBlocks, 256>>>(
                    d_C.get(), elements, 25);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(uint16_t) * config.numIterations;
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
        return "UINT16 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity UINT16 matrix multiplication and integer operations";
    }
    
    bool isSupported(int deviceId) const override {
        return true; // Integer operations are supported on all GPUs
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        return 3 * config.matrixSize * config.matrixSize * sizeof(uint16_t);
    }
};

// Register kernels
REGISTER_KERNEL(UINT8Kernel)
REGISTER_KERNEL(UINT16Kernel)