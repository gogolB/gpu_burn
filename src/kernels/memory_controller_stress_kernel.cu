#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Memory Controller Stress Kernel - Targets memory subsystem vulnerabilities
// Implements row hammer patterns, bank conflicts, and mixed granularity access

// Row hammer pattern - repeatedly access same memory rows
__global__ void row_hammer_kernel(float* memory, int rows, int cols, int hammer_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Target specific rows for hammering
    int target_row = tid % 2; // Alternate between adjacent rows
    int aggressor_row = target_row ^ 1; // XOR to get adjacent row
    
    // Hammer pattern: rapidly alternate between rows
    for (int iter = 0; iter < hammer_iterations; iter++) {
        // Read from aggressor row (causes activation)
        float val = memory[aggressor_row * cols + (tid % cols)];
        
        // Memory barrier to ensure completion
        __threadfence();
        
        // Write to target row (potential victim)
        memory[target_row * cols + (tid % cols)] = val * 1.0001f;
        
        // Force cache flush
        __threadfence_system();
    }
}

// Bank conflict stress - force maximum bank conflicts
__global__ void bank_conflict_kernel(float* memory, int stride, int iterations) {
    __shared__ float shared_mem[1024];
    int tid = threadIdx.x;
    int bank_id = tid % 32; // 32 banks in modern GPUs
    
    // Create maximum bank conflicts by having all threads access same bank
    for (int iter = 0; iter < iterations; iter++) {
        // All threads in warp access same bank
        int conflict_idx = (bank_id * 32 + (iter % 32)) % 1024;
        shared_mem[conflict_idx] = float(tid * iter);
        __syncthreads();
        
        // Read with conflicts
        float val = shared_mem[conflict_idx];
        
        // Write to global memory with strided pattern
        int global_idx = blockIdx.x * blockDim.x + tid;
        memory[global_idx * stride + (iter % stride)] = val;
    }
}

// Mixed granularity access - stress memory controller with varying access sizes
__global__ void mixed_granularity_kernel(char* memory, size_t total_size, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    // Initialize random state
    curandState state;
    curand_init(tid * 1234 + clock(), tid, 0, &state);
    
    for (int iter = 0; iter < iterations; iter++) {
        // Randomly choose access granularity
        int granularity = 1 << (curand(&state) % 7); // 1, 2, 4, 8, 16, 32, 64 bytes
        
        // Calculate safe access position
        size_t max_offset = total_size - granularity;
        size_t offset = (size_t(curand(&state)) % max_offset) & ~(granularity - 1); // Align
        
        // Perform mixed access patterns
        switch (granularity) {
            case 1: // Byte access
                memory[offset] ^= 0xFF;
                break;
            case 2: // Short access
                *reinterpret_cast<short*>(&memory[offset]) += 1;
                break;
            case 4: // Int access
                atomicAdd(reinterpret_cast<int*>(&memory[offset]), 1);
                break;
            case 8: // Long access
                atomicAdd(reinterpret_cast<unsigned long long*>(&memory[offset]), 1);
                break;
            case 16: { // 128-bit access
                float4* ptr = reinterpret_cast<float4*>(&memory[offset]);
                float4 val = *ptr;
                val.x *= 1.0001f; val.y *= 1.0001f;
                val.z *= 1.0001f; val.w *= 1.0001f;
                *ptr = val;
                break;
            }
            case 32: { // 256-bit access (2x float4)
                float4* ptr32 = reinterpret_cast<float4*>(&memory[offset]);
                float4 val1 = ptr32[0];
                float4 val2 = ptr32[1];
                val1.x += 0.001f; val2.x += 0.001f;
                ptr32[0] = val1; ptr32[1] = val2;
                break;
            }
            case 64: { // 512-bit access (4x float4)
                float4* ptr64 = reinterpret_cast<float4*>(&memory[offset]);
                for (int i = 0; i < 4; i++) {
                    float4 v = ptr64[i];
                    v.x = fmaf(v.x, 1.001f, 0.001f);
                    ptr64[i] = v;
                }
                break;
            }
        }
        
        // Force memory fence to stress controller
        __threadfence_system();
    }
}

// ECC stress pattern - create patterns that stress ECC mechanisms
__global__ void ecc_stress_kernel(unsigned int* memory, size_t elements, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int idx = tid; idx < elements; idx += stride) {
        unsigned int val = memory[idx];
        
        for (int iter = 0; iter < iterations; iter++) {
            // Create patterns that stress ECC
            // Single bit flips
            val ^= (1U << (iter % 32));
            
            // Double bit flips (harder for ECC)
            if (iter % 4 == 0) {
                val ^= (3U << ((iter / 4) % 31));
            }
            
            // Burst errors (even harder for ECC)
            if (iter % 8 == 0) {
                val ^= (0xFU << ((iter / 8) % 28));
            }
            
            // Write back with atomic to ensure visibility
            atomicExch(&memory[idx], val);
            
            // Read back to verify
            unsigned int readback = memory[idx];
            if (readback != val) {
                // Potential ECC correction or failure
                atomicAdd(&memory[0], 1); // Count errors
            }
        }
    }
}

// Memory Controller Stress Kernel Implementation
class MemoryControllerStressKernel : public TypedKernel<float> {
private:
    char* d_mixed_memory_ = nullptr;
    unsigned int* d_ecc_memory_ = nullptr;
    size_t mixed_memory_size_ = 0;
    
public:
    ~MemoryControllerStressKernel() {
        if (d_mixed_memory_) cudaFree(d_mixed_memory_);
        if (d_ecc_memory_) cudaFree(d_ecc_memory_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Calculate memory requirements
            size_t matrix_elements = config.matrixSize * config.matrixSize;
            size_t memory_size = matrix_elements * sizeof(float);
            mixed_memory_size_ = memory_size * 2; // Extra space for mixed access
            
            // Allocate device memory
            DeviceBuffer<float> d_memory(matrix_elements);
            CUDA_CHECK(cudaMalloc(&d_mixed_memory_, mixed_memory_size_));
            CUDA_CHECK(cudaMalloc(&d_ecc_memory_, matrix_elements * sizeof(unsigned int)));
            
            // Initialize memory
            CUDA_CHECK(cudaMemset(d_memory.get(), 0, memory_size));
            CUDA_CHECK(cudaMemset(d_mixed_memory_, 0, mixed_memory_size_));
            CUDA_CHECK(cudaMemset(d_ecc_memory_, 0, matrix_elements * sizeof(unsigned int)));
            
            // Configure kernel launches
            int threads_per_block = 256;
            int num_blocks = (matrix_elements + threads_per_block - 1) / threads_per_block;
            
            CudaTimer timer;
            timer.start();
            
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Phase 1: Row hammer pattern
                row_hammer_kernel<<<num_blocks, threads_per_block>>>(
                    d_memory.get(), config.matrixSize, config.matrixSize, 1000);
                
                // Phase 2: Bank conflict stress
                bank_conflict_kernel<<<num_blocks, threads_per_block>>>(
                    d_memory.get(), 17, 100); // Prime number stride for conflicts
                
                // Phase 3: Mixed granularity access
                mixed_granularity_kernel<<<num_blocks * 2, threads_per_block>>>(
                    d_mixed_memory_, mixed_memory_size_, 50);
                
                // Phase 4: ECC stress
                ecc_stress_kernel<<<num_blocks, threads_per_block>>>(
                    d_ecc_memory_, matrix_elements, 20);
                
                // Synchronize and check for errors
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Validation callback if enabled
                if (config.validationCallback && iter % 10 == 0) {
                    config.validationCallback(iter, d_memory.get(), matrix_elements);
                }
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Store output for validation
            setOutputData(d_memory.get(), matrix_elements);
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate memory operations
            double total_memory_ops = config.numIterations * (
                matrix_elements * 2 * 1000 +  // Row hammer
                matrix_elements * 2 * 100 +   // Bank conflicts
                mixed_memory_size_ * 50 +     // Mixed granularity
                matrix_elements * 4 * 20      // ECC stress
            );
            
            result.memoryBandwidthGBps = (total_memory_ops / 1e9) / (result.executionTimeMs / 1000.0);
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
        return "Memory Controller Stress Kernel";
    }
    
    std::string getDescription() const override {
        return "Stresses memory controller with row hammer, bank conflicts, mixed granularity access, and ECC patterns";
    }
    
    bool isSupported(int deviceId) const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        return prop.major >= 5; // Requires compute capability 5.0+
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        size_t base = config.matrixSize * config.matrixSize;
        return base * sizeof(float) +      // Main memory
               base * 2 +                   // Mixed granularity buffer  
               base * sizeof(unsigned int); // ECC stress buffer
    }
};

REGISTER_KERNEL(MemoryControllerStressKernel)