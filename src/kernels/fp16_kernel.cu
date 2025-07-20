#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

// FP16 Matrix multiplication kernel using shared memory
template<int TILE_SIZE>
__global__ void matmul_fp16_kernel(const __half* __restrict__ A, 
                                   const __half* __restrict__ B, 
                                   __half* __restrict__ C, 
                                   int N) {
    __shared__ __half As[TILE_SIZE][TILE_SIZE];
    __shared__ __half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    __half sum = __float2half(0.0f);
    
    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < N && t * TILE_SIZE + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }
        
        if (col < N && t * TILE_SIZE + ty < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __hadd(sum, __hmul(As[ty][k], Bs[k][tx]));
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Tensor Core WMMA kernel for FP16
#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;

template<int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmma_fp16_kernel(const __half* __restrict__ A,
                                 const __half* __restrict__ B,
                                 __half* __restrict__ C,
                                 int M, int N, int K) {
    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }
    
    // Store the output
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}
#endif

// High-intensity FP16 kernel that stresses the GPU
__global__ void stress_fp16_kernel(__half* data, int N, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        __half val = data[idx];
        
        // Perform intensive computations
        for (int i = 0; i < iterations; i++) {
            val = __hadd(val, __float2half(0.001f));
            val = __hmul(val, __float2half(1.001f));
            val = __hdiv(val, __float2half(1.0001f));
            
            // Convert to float for transcendental functions
            float fval = __half2float(val);
            fval = sqrtf(fabsf(fval));
            fval = sinf(fval * 0.1f);
            fval = cosf(fval * 0.1f);
            val = __float2half(fval);
        }
        
        data[idx] = val;
    }
}

// FP16 kernel implementation
class FP16Kernel : public TypedKernel<__half> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    FP16Kernel() {
        cublasCreate(&cublasHandle_);
    }
    
    ~FP16Kernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Check FP16 support
            GpuInfo info = getGpuInfo(config.deviceId);
            if (!info.supportsFP16()) {
                result.success = false;
                result.errorMessage = "FP16 not supported on this GPU";
                return result;
            }
            
            size_t N = config.matrixSize;
            size_t elements = N * N;
            
            // Allocate device memory
            DeviceBuffer<__half> d_A(elements);
            DeviceBuffer<__half> d_B(elements);
            DeviceBuffer<__half> d_C(elements);
            
            // Initialize matrices
            __half* h_A = new __half[elements];
            __half* h_B = new __half[elements];
            
            for (size_t i = 0; i < elements; i++) {
                h_A[i] = __float2half(1.0f + (i % 100) * 0.01f);
                h_B[i] = __float2half(2.0f - (i % 100) * 0.01f);
            }
            
            copyToDevice(d_A.get(), h_A, elements);
            copyToDevice(d_B.get(), h_B, elements);
            
            // Start timing
            CudaTimer timer;
            timer.start();
            
            // Run iterations
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                if (config.useTensorCores && info.supportsTensorCores()) {
                    // Use cuBLAS for tensor core acceleration
                    __half alpha = __float2half(1.0f);
                    __half beta = __float2half(0.0f);
                    cublasHgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N, N, &alpha,
                               d_B.get(), N,
                               d_A.get(), N,
                               &beta,
                               d_C.get(), N);
                } else {
                    // Use custom kernel
                    const int TILE_SIZE = 16;
                    dim3 block(TILE_SIZE, TILE_SIZE);
                    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
                    matmul_fp16_kernel<TILE_SIZE><<<grid, block>>>(
                        d_A.get(), d_B.get(), d_C.get(), N);
                }
                
                // Also run stress kernel
                int stressBlocks = (elements + 255) / 256;
                stress_fp16_kernel<<<stressBlocks, 256>>>(d_C.get(), elements, 20);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            double totalOps = 2.0 * N * N * N * config.numIterations;
            result.gflops = (totalOps / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth (rough estimate)
            double totalBytes = 3.0 * elements * sizeof(__half) * config.numIterations;
            result.memoryBandwidthGBps = (totalBytes / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Set output data for validation
            setOutputData(d_C.get(), elements);
            
            // Perform validation if enabled
            performValidation(result, config);
            
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
        return "FP16 Burn Kernel";
    }
    
    std::string getDescription() const override {
        return "High-intensity FP16 matrix multiplication with tensor core support";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        return info.supportsFP16();
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // 3 matrices (A, B, C) of size N x N
        return 3 * config.matrixSize * config.matrixSize * sizeof(__half);
    }
};

// Register kernel
REGISTER_KERNEL(FP16Kernel)