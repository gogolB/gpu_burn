#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// Conditionally include tensor core support
#if __CUDA_ARCH__ >= 700
#include <mma.h>
#endif

// Mixed Precision Chaos Kernel - Stresses format conversion and mixed precision paths
// Implements concurrent FP64/FP32/FP16/INT8 operations, tensor core interleaving

// Concurrent mixed precision operations - stress different execution units
__global__ void concurrent_mixed_precision_kernel(
    double* fp64_data,
    float* fp32_data, 
    half* fp16_data,
    int8_t* int8_data,
    size_t elements,
    int iterations) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        // Initialize local variables of different types
        double d_val = fp64_data[tid];
        float f_val = fp32_data[tid];
        half h_val = fp16_data[tid];
        int8_t i_val = int8_data[tid];
        
        // Perform mixed precision operations concurrently
        for (int iter = 0; iter < iterations; iter++) {
            // FP64 operations
            d_val = sqrt(d_val * d_val + 1.0);
            d_val = exp(d_val * 0.0001);
            
            // FP32 operations 
            f_val = sqrtf(f_val * f_val + 1.0f);
            f_val = expf(f_val * 0.001f);
            
            // FP16 operations
            h_val = __hadd(h_val, __float2half(0.001f));
            h_val = __hmul(h_val, __float2half(1.001f));
            h_val = hsqrt(h_val);
            
            // INT8 operations
            i_val = (i_val * 3 + 7) % 127;
            
            // Cross-precision operations (stress conversion units)
            if (iter % 4 == 0) {
                // FP64 -> FP32 -> FP16 -> INT8
                f_val = float(d_val);
                h_val = __float2half(f_val);
                i_val = int8_t(__half2float(h_val) * 127.0f);
            } else if (iter % 4 == 1) {
                // INT8 -> FP16 -> FP32 -> FP64
                h_val = __float2half(float(i_val) / 127.0f);
                f_val = __half2float(h_val);
                d_val = double(f_val);
            } else if (iter % 4 == 2) {
                // Mixed operations
                d_val += double(__half2float(h_val));
                f_val *= float(i_val) / 127.0f;
                h_val = __float2half(float(d_val - floor(d_val)));
            }
            
            // Memory fence to ensure completion
            __threadfence();
        }
        
        // Write back results
        fp64_data[tid] = d_val;
        fp32_data[tid] = f_val;
        fp16_data[tid] = h_val;
        int8_data[tid] = i_val;
    }
}

// Format conversion stress - rapid conversions between formats
__global__ void format_conversion_stress_kernel(
    float* workspace,
    size_t elements,
    int iterations) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        float val = workspace[tid];
        
        for (int iter = 0; iter < iterations; iter++) {
            // Rapid format conversions to stress conversion units
            
            // Float -> Half -> BFloat16 -> Float
            half h = __float2half(val);
            __nv_bfloat16 bf = __float2bfloat16(val);
            val = __half2float(h) + __bfloat162float(bf);
            
            // Float -> Double -> Float
            double d = double(val);
            d = sqrt(d * d + 1.0);
            val = float(d);
            
            // Float -> Int -> Float (with saturation)
            int i = __float2int_rn(val * 1000.0f);
            val = float(i) / 1000.0f;
            
            // Float -> Unsigned -> Float
            unsigned int u = __float2uint_rn(fabsf(val) * 1000.0f);
            val = float(u) / 1000.0f;
            
            // Multiple chained conversions
            h = __float2half(val);
            bf = __float2bfloat16(__half2float(h));
            d = double(__bfloat162float(bf));
            val = float(d);
            
            // Stress denormal handling
            if (iter % 10 == 0) {
                val *= 1e-38f; // Create denormals
            } else if (iter % 10 == 1) {
                val *= 1e38f; // Return to normal range
            }
        }
        
        workspace[tid] = val;
    }
}

// Tensor core interleaving - mix tensor core and regular operations
__global__ void tensor_core_interleave_kernel(
    half* a_matrix,
    half* b_matrix,
    float* c_matrix,
    float* d_regular,
    int M, int N, int K,
    int iterations) {
    
#if __CUDA_ARCH__ >= 700
    // Tensor core fragment declarations
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x % 32;
    
    // Declare fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    for (int iter = 0; iter < iterations; iter++) {
        // Phase 1: Tensor core operations
        if (warpId * WMMA_M < M && warpId * WMMA_N < N) {
            // Load matrices
            nvcuda::wmma::load_matrix_sync(a_frag, a_matrix + warpId * WMMA_M * K, K);
            nvcuda::wmma::load_matrix_sync(b_frag, b_matrix + warpId * WMMA_K * N, N);
            
            // Perform matrix multiply-accumulate
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        
        // Phase 2: Interleave with regular FP32 operations
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < M * N) {
            float val = d_regular[tid];
            
            // Regular precision operations
            val = sqrtf(val * val + 1.0f);
            val = expf(val * 0.001f);
            val = sinf(val);
            
            // Synchronize within warp
            __syncwarp();
            
            // Mix with tensor core results (causes pipeline stalls)
            if (laneId == 0 && warpId * WMMA_M < M && warpId * WMMA_N < N) {
                // Store tensor core results
                nvcuda::wmma::store_matrix_sync(c_matrix + warpId * WMMA_M * N + warpId * WMMA_N, c_frag, N, nvcuda::wmma::mem_row_major);
            }
            
            // More regular operations
            val = cosf(val);
            val = fmaf(val, 1.001f, 0.001f);
            
            d_regular[tid] = val;
        }
        
        // Force synchronization to stress scheduling
        __syncthreads();
    }
#else
    // Fallback for older GPUs - just do regular operations
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M * N) {
        float val = d_regular[tid];
        
        for (int iter = 0; iter < iterations; iter++) {
            // Simulate mixed precision workload without tensor cores
            half h_val = __float2half(val);
            
            // Convert through multiple precisions
            val = sqrtf(val * val + 1.0f);
            h_val = hsqrt(__hadd(h_val, __float2half(1.0f)));
            val = __half2float(h_val) + val;
            
            val = expf(val * 0.001f);
            val = sinf(val);
            val = cosf(val);
            val = fmaf(val, 1.001f, 0.001f);
        }
        
        d_regular[tid] = val;
        c_matrix[tid] = val; // Write something to c_matrix
    }
#endif
}

// Race condition stress - create precision-dependent race conditions
__global__ void precision_race_kernel(
    float* shared_buffer,
    half* half_buffer,
    double* double_buffer,
    size_t elements,
    int iterations) {
    
    __shared__ float s_float[256];
    __shared__ half s_half[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // Initialize shared memory
    if (local_tid < 256) {
        s_float[local_tid] = shared_buffer[tid % elements];
        s_half[local_tid] = half_buffer[tid % elements];
    }
    __syncthreads();
    
    for (int iter = 0; iter < iterations; iter++) {
        // Create race conditions between different precision operations
        
        // Thread 0-63: Update float values
        if (local_tid < 64) {
            float val = s_float[local_tid];
            val = sqrtf(val * val + 1.0f);
            s_float[local_tid] = val;
            
            // Also read/write adjacent half values (potential race)
            s_half[local_tid + 64] = __float2half(val);
        }
        
        // Thread 64-127: Update half values
        else if (local_tid < 128) {
            half val = s_half[local_tid];
            val = hsqrt(__hadd(val, __float2half(1.0f)));
            s_half[local_tid] = val;
            
            // Also read/write adjacent float values (potential race)
            s_float[local_tid - 64] = __half2float(val);
        }
        
        // Thread 128-255: Mixed operations creating races
        else {
            int idx1 = (local_tid * 7) % 256;
            int idx2 = (local_tid * 13) % 256;
            
            // Non-atomic mixed precision operations (races)
            float f_val = s_float[idx1] + __half2float(s_half[idx2]);
            half h_val = __float2half(s_float[idx2]) + s_half[idx1];
            
            s_float[idx2] = f_val;
            s_half[idx1] = h_val;
            
            // Double precision races with global memory
            if (tid < elements) {
                double d_val = double_buffer[tid];
                d_val += double(f_val) + double(__half2float(h_val));
                double_buffer[tid] = d_val;
            }
        }
        
        // Partial synchronization (creates more races)
        if (iter % 3 == 0) {
            __syncthreads();
        }
    }
    
    // Write back results
    if (tid < elements && local_tid < 256) {
        shared_buffer[tid] = s_float[local_tid];
        half_buffer[tid] = s_half[local_tid];
    }
}

// Mixed Precision Chaos Kernel Implementation
class MixedPrecisionChaosKernel : public TypedKernel<float> {
private:
    double* d_fp64_data_ = nullptr;
    half* d_fp16_data_ = nullptr;
    int8_t* d_int8_data_ = nullptr;
    double* d_double_buffer_ = nullptr;
    half* d_half_buffer_ = nullptr;
    
public:
    ~MixedPrecisionChaosKernel() {
        if (d_fp64_data_) cudaFree(d_fp64_data_);
        if (d_fp16_data_) cudaFree(d_fp16_data_);
        if (d_int8_data_) cudaFree(d_int8_data_);
        if (d_double_buffer_) cudaFree(d_double_buffer_);
        if (d_half_buffer_) cudaFree(d_half_buffer_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Check for required capabilities
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, config.deviceId));
            
            if (prop.major < 7) {
                result.success = false;
                result.errorMessage = "Mixed precision chaos kernel requires compute capability 7.0+";
                return result;
            }
            
            // Calculate memory requirements
            size_t elements = config.matrixSize * config.matrixSize;
            
            // Allocate buffers for different precisions
            DeviceBuffer<float> d_fp32_data(elements);
            CUDA_CHECK(cudaMalloc(&d_fp64_data_, elements * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_fp16_data_, elements * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_int8_data_, elements * sizeof(int8_t)));
            CUDA_CHECK(cudaMalloc(&d_double_buffer_, elements * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_half_buffer_, elements * sizeof(half)));
            
            // Initialize data
            float* h_data = new float[elements];
            for (size_t i = 0; i < elements; i++) {
                h_data[i] = 1.0f + (i % 100) * 0.01f;
            }
            
            copyToDevice(d_fp32_data.get(), h_data, elements);
            
            // Initialize other precision buffers
            CUDA_CHECK(cudaMemcpy(d_fp64_data_, h_data, elements * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_fp16_data_, h_data, elements * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_int8_data_, 1, elements * sizeof(int8_t)));
            
            delete[] h_data;
            
            // Configure kernel launches
            int threads_per_block = 256;
            int num_blocks = (elements + threads_per_block - 1) / threads_per_block;
            
            // For tensor core kernel
            int M = config.matrixSize;
            int N = config.matrixSize;
            int K = 256; // Small K for more frequent tensor core calls
            
            CudaTimer timer;
            timer.start();
            
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Phase 1: Concurrent mixed precision operations
                concurrent_mixed_precision_kernel<<<num_blocks, threads_per_block>>>(
                    d_fp64_data_, d_fp32_data.get(), d_fp16_data_, d_int8_data_,
                    elements, 20);
                
                // Phase 2: Format conversion stress
                format_conversion_stress_kernel<<<num_blocks, threads_per_block>>>(
                    d_fp32_data.get(), elements, 50);
                
                // Phase 3: Tensor core interleaving (if supported)
                if (config.useTensorCores && prop.major >= 7) {
                    // Ensure proper dimensions for tensor cores
                    int tc_blocks = (M * N / 256 + 255) / 256;
                    tensor_core_interleave_kernel<<<tc_blocks, 256>>>(
                        d_fp16_data_, d_fp16_data_ + K * N,
                        d_fp32_data.get(), d_fp32_data.get() + M * N / 2,
                        M, N, K, 10);
                }
                
                // Phase 4: Precision race conditions
                precision_race_kernel<<<num_blocks, threads_per_block>>>(
                    d_fp32_data.get(), d_half_buffer_, d_double_buffer_,
                    elements, 30);
                
                // Synchronize
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Validation callback if enabled
                if (config.validationCallback && iter % 10 == 0) {
                    config.validationCallback(iter, d_fp32_data.get(), elements);
                }
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Store output for validation
            setOutputData(d_fp32_data.get(), elements);
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate operations (mixed precision makes this complex)
            double total_ops = config.numIterations * elements * (
                20 * 10 +   // Concurrent mixed precision
                50 * 8 +    // Format conversions
                10 * 16 +   // Tensor cores (if used)
                30 * 5      // Race operations
            );
            
            result.gflops = (total_ops / 1e9) / (result.executionTimeMs / 1000.0);
            result.success = true;
            
            // Perform validation if enabled
            performValidation(result, config);
            
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = e.what();
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Mixed Precision Chaos Kernel";
    }
    
    std::string getDescription() const override {
        return "Stresses mixed precision execution paths, format conversions, tensor core interleaving, and precision-dependent races";
    }
    
    bool isSupported(int deviceId) const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        return prop.major >= 6; // FP16 support, better with 7.0+ for tensor cores
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        size_t elements = config.matrixSize * config.matrixSize;
        return elements * (sizeof(float) + sizeof(double) + sizeof(half) + 
                          sizeof(int8_t) + sizeof(double) + sizeof(half));
    }
};

REGISTER_KERNEL(MixedPrecisionChaosKernel)