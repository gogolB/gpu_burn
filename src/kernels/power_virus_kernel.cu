#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Power Virus Kernel - Creates maximum instantaneous power draw patterns
// Designed to stress power delivery, VRMs, and expose power-related faults

// Maximum ALU utilization - all functional units active
__global__ void max_alu_stress_kernel(float* data, int elements, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        float val1 = data[tid];
        float val2 = val1 * 1.1f;
        float val3 = val1 * 1.2f;
        float val4 = val1 * 1.3f;
        float val5 = val1 * 1.4f;
        float val6 = val1 * 1.5f;
        float val7 = val1 * 1.6f;
        float val8 = val1 * 1.7f;
        
        // Unrolled loop for maximum ILP (Instruction Level Parallelism)
        for (int i = 0; i < iterations; i++) {
            // Wave 1: Transcendental functions (highest power draw)
            val1 = __powf(val1, 1.0001f);
            val2 = __expf(val2 * 0.001f);
            val3 = __logf(fabsf(val3) + 1.0f);
            val4 = sqrtf(fabsf(val4) + 0.001f);
            val5 = rsqrtf(fabsf(val5) + 1.0f);
            val6 = sinf(val6);
            val7 = cosf(val7);
            val8 = tanf(val8 * 0.1f);
            
            // Wave 2: More transcendentals
            val1 = atanf(val1);
            val2 = atan2f(val2, val3);
            val3 = sinhf(val3 * 0.1f);
            val4 = coshf(val4 * 0.1f);
            val5 = tanhf(val5 * 0.1f);
            val6 = asinhf(val6 * 0.1f);
            val7 = acoshf(val7 + 1.1f);
            val8 = atanhf(val8 * 0.1f);
            
            // Wave 3: Special functions
            val1 = erfcf(val1 * 0.01f);
            val2 = erfinvf(val2 * 0.1f);
            val3 = normcdff(val3 * 0.1f);
            val4 = normcdfinvf(val4 * 0.1f);
            val5 = lgammaf(fabsf(val5) + 1.0f);
            val6 = tgammaf(fabsf(val6 * 0.1f) + 1.0f);
            val7 = j0f(val7);
            val8 = y0f(fabsf(val8) + 1.0f);
            
            // Wave 4: FMA operations (use FMA units)
            val1 = fmaf(val1, 1.001f, val2);
            val2 = fmaf(val2, 1.001f, val3);
            val3 = fmaf(val3, 1.001f, val4);
            val4 = fmaf(val4, 1.001f, val5);
            val5 = fmaf(val5, 1.001f, val6);
            val6 = fmaf(val6, 1.001f, val7);
            val7 = fmaf(val7, 1.001f, val8);
            val8 = fmaf(val8, 1.001f, val1);
            
            // Wave 5: Division (power hungry)
            val1 = __fdividef(val1, 1.0001f);
            val2 = __fdividef(val2, 1.0002f);
            val3 = __fdividef(val3, 1.0003f);
            val4 = __fdividef(val4, 1.0004f);
            val5 = __fdividef(val5, 1.0005f);
            val6 = __fdividef(val6, 1.0006f);
            val7 = __fdividef(val7, 1.0007f);
            val8 = __fdividef(val8, 1.0008f);
        }
        
        // Combine results to prevent optimization
        data[tid] = val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8;
    }
}

// Memory bandwidth saturation - maximum memory traffic
__global__ void max_memory_bandwidth_kernel(float* src, float* dst, int elements, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Use vector loads/stores for maximum bandwidth
    for (int iter = 0; iter < iterations; iter++) {
        // Coalesced vector loads and stores
        for (int idx = tid; idx < elements/4; idx += stride) {
            float4 val = reinterpret_cast<float4*>(src)[idx];
            
            // Small computation to prevent optimization
            val.x *= 1.0001f;
            val.y *= 1.0001f;
            val.z *= 1.0001f;
            val.w *= 1.0001f;
            
            reinterpret_cast<float4*>(dst)[idx] = val;
        }
        
        // Swap pointers for next iteration
        float* temp = src;
        src = dst;
        dst = temp;
    }
}

// Combined ALU + Memory stress - maximum total power
__global__ void combined_power_stress_kernel(
    float* memory, 
    float* workspace,
    int elements, 
    int compute_iterations,
    int memory_iterations) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_data[1024];
    
    if (tid < elements) {
        float reg[8];
        
        // Load initial values
        for (int i = 0; i < 8; i++) {
            reg[i] = memory[(tid + i * 1024) % elements];
        }
        
        // Alternate between compute and memory phases
        for (int phase = 0; phase < memory_iterations; phase++) {
            // Compute phase - stress ALUs
            for (int iter = 0; iter < compute_iterations; iter++) {
                // Unrolled transcendental operations
                reg[0] = __powf(reg[0], 1.0001f);
                reg[1] = __expf(reg[1] * 0.001f);
                reg[2] = sqrtf(fabsf(reg[2]) + 0.001f);
                reg[3] = sinf(reg[3]);
                reg[4] = cosf(reg[4]);
                reg[5] = tanf(reg[5] * 0.1f);
                reg[6] = atanf(reg[6]);
                reg[7] = erfcf(reg[7] * 0.01f);
            }
            
            // Memory phase - stress memory subsystem
            int local_tid = threadIdx.x;
            
            // Shared memory traffic
            shared_data[local_tid] = reg[0];
            shared_data[(local_tid + 256) % 1024] = reg[1];
            shared_data[(local_tid + 512) % 1024] = reg[2];
            shared_data[(local_tid + 768) % 1024] = reg[3];
            __syncthreads();
            
            // Global memory traffic with atomics
            for (int i = 0; i < 4; i++) {
                int idx = (tid * 4 + i + phase * 1024) % elements;
                atomicAdd(&workspace[idx], reg[i]);
                reg[i+4] += workspace[(idx + elements/2) % elements];
            }
            
            // Texture memory access (if available)
            // Additional memory patterns
            if (phase % 2 == 0) {
                // Strided access pattern
                for (int i = 0; i < 8; i++) {
                    memory[(tid * 17 + i * 1024) % elements] = reg[i];
                }
            } else {
                // Random access pattern
                curandState state;
                curand_init(tid + phase, tid, 0, &state);
                for (int i = 0; i < 8; i++) {
                    int rand_idx = curand(&state) % elements;
                    reg[i] += memory[rand_idx] * 0.0001f;
                }
            }
        }
        
        // Store final results
        for (int i = 0; i < 8; i++) {
            memory[(tid + i * 1024) % elements] = reg[i];
        }
    }
}

// Instantaneous power spike pattern
__global__ void power_spike_kernel(float* data, int elements, int spike_duration, int idle_duration) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        float val = data[tid];
        
        // Spike phase - maximum power consumption
        for (int i = 0; i < spike_duration; i++) {
            // All units active simultaneously
            float temp1 = __powf(val, 1.0001f);
            float temp2 = __expf(val * 0.001f);
            float temp3 = sqrtf(fabsf(val) + 0.001f);
            float temp4 = sinf(val);
            float temp5 = cosf(val);
            float temp6 = __fdividef(val, 1.0001f);
            
            // Force dependency chain
            val = temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
            
            // Memory operations
            atomicAdd(&data[(tid + i) % elements], val * 0.00001f);
        }
        
        // Idle phase - minimal activity
        for (int i = 0; i < idle_duration; i++) {
            val = val * 1.0000001f; // Minimal operation
        }
        
        data[tid] = val;
    }
}

// Voltage droop inducer - rapid power transitions
__global__ void voltage_droop_kernel(float* data, int elements, int pattern_type) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        float val = data[tid];
        
        switch (pattern_type) {
            case 0: // Square wave pattern
                for (int cycle = 0; cycle < 100; cycle++) {
                    if (cycle % 2 == 0) {
                        // High power
                        for (int i = 0; i < 10; i++) {
                            val = __powf(val, 1.0001f);
                            val = __expf(val * 0.001f);
                            val = sqrtf(fabsf(val) + 0.001f);
                        }
                    } else {
                        // Low power
                        val = val * 1.0001f;
                    }
                }
                break;
                
            case 1: // Sawtooth pattern
                for (int cycle = 0; cycle < 100; cycle++) {
                    int intensity = cycle % 10;
                    for (int i = 0; i < intensity; i++) {
                        val = __powf(val, 1.0001f);
                        val = sinf(val);
                    }
                }
                break;
                
            case 2: // Random transitions
                curandState state;
                curand_init(tid, tid, 0, &state);
                for (int cycle = 0; cycle < 100; cycle++) {
                    int intensity = curand(&state) % 20;
                    for (int i = 0; i < intensity; i++) {
                        val = __expf(val * 0.001f);
                        val = cosf(val);
                    }
                }
                break;
                
            case 3: // Resonance pattern (match VRM frequency)
                for (int cycle = 0; cycle < 1000; cycle++) {
                    // ~300kHz switching frequency simulation
                    if ((cycle % 3) == 0) {
                        val = __powf(val, 1.0001f);
                        val = __fdividef(val, 1.0001f);
                    } else {
                        val *= 1.00001f;
                    }
                }
                break;
        }
        
        data[tid] = val;
    }
}

// Power Virus Kernel Implementation
class PowerVirusKernel : public TypedKernel<float> {
private:
    float* d_workspace_ = nullptr;
    
public:
    ~PowerVirusKernel() {
        if (d_workspace_) cudaFree(d_workspace_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Get device properties for power limits
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, config.deviceId));
            
            // Calculate memory requirements
            size_t elements = config.matrixSize * config.matrixSize;
            size_t memory_size = elements * sizeof(float);
            
            // Allocate device memory
            DeviceBuffer<float> d_primary(elements);
            DeviceBuffer<float> d_secondary(elements);
            CUDA_CHECK(cudaMalloc(&d_workspace_, memory_size));
            
            // Initialize with patterns that prevent optimization
            float* h_data = new float[elements];
            for (size_t i = 0; i < elements; i++) {
                h_data[i] = 1.0f + sinf(float(i) * 0.001f);
            }
            copyToDevice(d_primary.get(), h_data, elements);
            copyToDevice(d_secondary.get(), h_data, elements);
            delete[] h_data;
            
            // Configure for maximum occupancy and power draw
            int threads_per_block = 256;
            int num_blocks = prop.multiProcessorCount * 32; // Oversubscribe SMs
            
            CudaTimer timer;
            timer.start();
            
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Phase 1: Maximum ALU stress
                max_alu_stress_kernel<<<num_blocks, threads_per_block>>>(
                    d_primary.get(), elements, 50);
                
                // Phase 2: Maximum memory bandwidth
                max_memory_bandwidth_kernel<<<num_blocks * 2, threads_per_block>>>(
                    d_primary.get(), d_secondary.get(), elements, 10);
                
                // Phase 3: Combined ALU + Memory stress
                combined_power_stress_kernel<<<num_blocks, threads_per_block>>>(
                    d_primary.get(), d_workspace_, elements, 20, 5);
                
                // Phase 4: Power spike patterns
                int spike_pattern = iter % 4;
                power_spike_kernel<<<num_blocks, threads_per_block>>>(
                    d_primary.get(), elements, 50 + spike_pattern * 10, 10);
                
                // Phase 5: Voltage droop patterns
                int droop_pattern = (iter / 4) % 4;
                voltage_droop_kernel<<<num_blocks, threads_per_block>>>(
                    d_secondary.get(), elements, droop_pattern);
                
                // Synchronize to ensure all work completes
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Validation callback if enabled
                if (config.validationCallback && iter % 5 == 0) {
                    config.validationCallback(iter, d_primary.get(), elements);
                }
                
                // Monitoring callback for power tracking
                if (config.monitoringCallback && iter % 2 == 0) {
                    KernelResult interim_result;
                    interim_result.success = true;
                    interim_result.executionTimeMs = timer.getElapsedMs();
                    // Note: Real implementation would query actual power here
                    interim_result.avgPowerWatts = 250.0 + (iter % 10) * 10.0;
                    config.monitoringCallback(iter, interim_result);
                }
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Store output for validation
            setOutputData(d_primary.get(), elements);
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate operations (very rough for power virus)
            double total_ops = config.numIterations * elements * (
                50 * 40 +   // ALU stress operations
                10 * 2 +    // Memory bandwidth operations
                20 * 8 +    // Combined stress operations
                50 * 6 +    // Power spike operations
                100 * 2     // Voltage droop operations
            );
            
            result.gflops = (total_ops / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Estimate power (this would need actual measurement)
            result.avgPowerWatts = 300.0; // Placeholder
            
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
        return "Power Virus Kernel";
    }
    
    std::string getDescription() const override {
        return "Maximum instantaneous power draw patterns to stress power delivery, VRMs, and induce voltage droops";
    }
    
    bool isSupported(int deviceId) const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        return prop.major >= 5; // Basic requirement
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        size_t elements = config.matrixSize * config.matrixSize;
        return 3 * elements * sizeof(float); // Primary, secondary, and workspace
    }
};

REGISTER_KERNEL(PowerVirusKernel)