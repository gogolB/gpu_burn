#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Thermal Gradient Kernel - Creates asymmetric thermal patterns to stress cooling
// Implements hotspot migration, rapid power cycling, and SM-specific loading

// Asymmetric SM loading - create hotspots by loading specific SMs heavily
__global__ void asymmetric_sm_load_kernel(float* data, int elements_per_sm, int target_sm_mask, int iterations) {
    // Get SM ID (device-specific, approximation)
    int sm_id = blockIdx.x % 108; // Assuming up to 108 SMs (A100)
    
    // Check if this SM should be heavily loaded
    bool heavy_load = (target_sm_mask >> (sm_id % 32)) & 1;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements_per_sm * gridDim.x / blockDim.x) {
        float val = data[tid];
        
        // Heavy computational load for selected SMs
        if (heavy_load) {
            // Maximum computational intensity
            for (int i = 0; i < iterations * 10; i++) {
                // Mix of power-hungry operations
                val = __powf(val, 1.0001f);
                val = __expf(val * 0.001f);
                val = __logf(fabsf(val) + 1.0f);
                val = sqrtf(fabsf(val));
                val = __fdividef(val, 1.0001f);
                val = fmaf(val, 1.001f, 0.001f);
                val = sinf(val);
                val = cosf(val);
                val = tanf(val * 0.1f);
                val = atanf(val);
                val = tanhf(val * 0.1f);
                val = erfcf(val * 0.001f);
            }
        } else {
            // Light load for other SMs
            for (int i = 0; i < iterations; i++) {
                val = val * 1.0001f + 0.0001f;
            }
        }
        
        data[tid] = val;
    }
}

// Hotspot migration - move thermal load across different GPU regions
__global__ void hotspot_migration_kernel(float* data, int region_size, int current_hotspot, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int region = tid / region_size;
    
    // Calculate distance from current hotspot
    int hotspot_distance = abs(region - current_hotspot);
    
    if (tid < region_size * 8) { // 8 thermal regions
        float val = data[tid];
        
        // Load intensity based on distance from hotspot
        int load_factor = (hotspot_distance == 0) ? 100 : 
                         (hotspot_distance == 1) ? 50 : 
                         (hotspot_distance == 2) ? 25 : 10;
        
        for (int i = 0; i < iterations * load_factor / 10; i++) {
            // Power-intensive operations
            val = __powf(val, 1.00001f);
            val = __expf(val * 0.0001f);
            val = sqrtf(fabsf(val) + 0.001f);
            val = __fdividef(val, 1.00001f);
            
            // Memory operations to increase power draw
            if (i % 10 == 0) {
                atomicAdd(&data[(tid + i) % (region_size * 8)], 0.0001f);
            }
        }
        
        data[tid] = val;
    }
}

// Rapid power cycling - alternate between high and low power states
__global__ void power_cycling_kernel(float* data, int elements, int cycle_phase, int high_power_iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < elements) {
        float val = data[tid];
        
        if (cycle_phase % 2 == 0) {
            // High power phase - maximum computational load
            for (int i = 0; i < high_power_iterations; i++) {
                // Double precision operations (if supported) for max power
                double dval = double(val);
                dval = pow(dval, 1.000001);
                dval = exp(dval * 0.00001);
                dval = log(abs(dval) + 1.0);
                dval = sqrt(abs(dval));
                val = float(dval);
                
                // Additional FP32 operations
                val = __powf(val, 1.0001f);
                val = __expf(val * 0.001f);
                val = erfcf(val * 0.001f);
                val = normcdff(val * 0.1f);
            }
            
            // Force all threads to participate in memory operations
            __shared__ float shared_data[256];
            shared_data[threadIdx.x] = val;
            __syncthreads();
            
            // Create memory traffic
            for (int i = 0; i < 32; i++) {
                val += shared_data[(threadIdx.x + i) % 256] * 0.0001f;
            }
            __syncthreads();
            
        } else {
            // Low power phase - minimal operations
            val = val * 1.0001f + 0.0001f;
            // Insert delay loop to reduce power
            for (int delay = 0; delay < 100; delay++) {
                val = fmaf(val, 1.0f, 0.0f); // NOP-like operation
            }
        }
        
        data[tid] = val;
    }
}

// Thermal stress pattern - create specific temperature gradients
__global__ void thermal_stress_pattern_kernel(float* data, int width, int height, int pattern_type, int iterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float val = data[idx];
        
        // Different thermal patterns
        float load_factor = 1.0f;
        
        switch (pattern_type) {
            case 0: // Checkerboard pattern
                load_factor = ((x / 32) + (y / 32)) % 2 ? 10.0f : 1.0f;
                break;
            case 1: // Diagonal gradient
                load_factor = float(x + y) / float(width + height) * 10.0f + 1.0f;
                break;
            case 2: // Center hotspot
                {
                    int cx = width / 2;
                    int cy = height / 2;
                    float dist = sqrtf(float((x-cx)*(x-cx) + (y-cy)*(y-cy)));
                    load_factor = fmaxf(1.0f, 10.0f - dist / 100.0f);
                }
                break;
            case 3: // Edge heating
                {
                    int edge_dist = min(min(x, width-x-1), min(y, height-y-1));
                    load_factor = edge_dist < 32 ? 10.0f : 1.0f;
                }
                break;
        }
        
        // Apply thermal load
        for (int i = 0; i < int(iterations * load_factor); i++) {
            val = __powf(val, 1.0001f);
            val = sqrtf(fabsf(val) + 0.001f);
            val = fmaf(val, 1.001f, 0.001f);
            
            // Occasional memory operations
            if (i % 100 == 0) {
                atomicAdd(&data[idx], 0.00001f);
            }
        }
        
        data[idx] = val;
    }
}

// Warp divergence thermal stress - create thermal variations within warps
__global__ void warp_divergence_thermal_kernel(float* data, int elements, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    
    if (tid < elements) {
        float val = data[tid];
        
        // Create divergent execution paths within warps
        if (lane_id < 8) {
            // Heavy compute path - 25% of threads
            for (int i = 0; i < iterations * 4; i++) {
                val = __powf(val, 1.0001f);
                val = __expf(val * 0.001f);
                val = erfcf(val * 0.001f);
            }
        } else if (lane_id < 16) {
            // Medium compute path - 25% of threads
            for (int i = 0; i < iterations * 2; i++) {
                val = sqrtf(fabsf(val) + 0.001f);
                val = __fdividef(val, 1.0001f);
            }
        } else if (lane_id < 24) {
            // Light compute path - 25% of threads
            for (int i = 0; i < iterations; i++) {
                val = fmaf(val, 1.001f, 0.001f);
            }
        } else {
            // Minimal compute path - 25% of threads
            val = val * 1.0001f + 0.0001f;
        }
        
        // Force warp synchronization
        __syncwarp();
        
        data[tid] = val;
    }
}

// Thermal Gradient Kernel Implementation
class ThermalGradientKernel : public TypedKernel<float> {
private:
    int current_hotspot_ = 0;
    int cycle_phase_ = 0;
    int pattern_type_ = 0;
    
public:
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // Get device properties
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, config.deviceId));
            int num_sms = prop.multiProcessorCount;
            
            // Calculate memory requirements
            size_t elements = config.matrixSize * config.matrixSize;
            size_t memory_size = elements * sizeof(float);
            
            // Allocate device memory
            DeviceBuffer<float> d_data(elements);
            
            // Initialize with pattern
            float* h_data = new float[elements];
            for (size_t i = 0; i < elements; i++) {
                h_data[i] = 1.0f + (i % 1000) * 0.001f;
            }
            copyToDevice(d_data.get(), h_data, elements);
            delete[] h_data;
            
            // Configure kernel launches
            dim3 block(256);
            dim3 grid((elements + block.x - 1) / block.x);
            
            dim3 block2d(16, 16);
            dim3 grid2d((config.matrixSize + block2d.x - 1) / block2d.x,
                       (config.matrixSize + block2d.y - 1) / block2d.y);
            
            CudaTimer timer;
            timer.start();
            
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Phase 1: Asymmetric SM loading
                // Create different SM load patterns
                int sm_mask = 0;
                if (iter % 10 < 3) {
                    sm_mask = 0x0000FFFF; // Load first half of SMs
                } else if (iter % 10 < 6) {
                    sm_mask = 0xFFFF0000; // Load second half of SMs
                } else {
                    sm_mask = 0x55555555; // Alternating pattern
                }
                
                asymmetric_sm_load_kernel<<<grid.x, block.x>>>(
                    d_data.get(), elements / num_sms, sm_mask, 10);
                
                // Phase 2: Hotspot migration
                current_hotspot_ = (iter / 5) % 8; // Move hotspot every 5 iterations
                hotspot_migration_kernel<<<grid.x, block.x>>>(
                    d_data.get(), elements / 8, current_hotspot_, 20);
                
                // Phase 3: Power cycling
                cycle_phase_ = iter % 4; // Cycle every 4 iterations
                power_cycling_kernel<<<grid.x, block.x>>>(
                    d_data.get(), elements, cycle_phase_, 50);
                
                // Phase 4: Thermal patterns
                pattern_type_ = (iter / 20) % 4; // Change pattern every 20 iterations
                thermal_stress_pattern_kernel<<<grid2d, block2d>>>(
                    d_data.get(), config.matrixSize, config.matrixSize, pattern_type_, 5);
                
                // Phase 5: Warp divergence thermal stress
                warp_divergence_thermal_kernel<<<grid.x, block.x>>>(
                    d_data.get(), elements, 30);
                
                // Synchronize
                CUDA_CHECK(cudaDeviceSynchronize());
                
                // Validation callback if enabled
                if (config.validationCallback && iter % 5 == 0) {
                    config.validationCallback(iter, d_data.get(), elements);
                }
                
                // Monitoring callback
                if (config.monitoringCallback && iter % 10 == 0) {
                    KernelResult interim_result;
                    interim_result.success = true;
                    interim_result.executionTimeMs = timer.getElapsedMs();
                    // Note: Real implementation would query actual temperature here
                    interim_result.avgTemperatureCelsius = 70.0 + (iter % 20);
                    config.monitoringCallback(iter, interim_result);
                }
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Store output for validation
            setOutputData(d_data.get(), elements);
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate operations
            double total_ops = config.numIterations * elements * (
                10 * 12 +  // Asymmetric SM ops
                20 * 4 +   // Hotspot migration ops
                50 * 10 +  // Power cycling ops
                5 * 3 +    // Thermal pattern ops
                30 * 3     // Warp divergence ops
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
        return "Thermal Gradient Kernel";
    }
    
    std::string getDescription() const override {
        return "Creates asymmetric thermal patterns, hotspot migration, and rapid power cycling to stress GPU cooling";
    }
    
    bool isSupported(int deviceId) const override {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        return prop.major >= 6; // Requires compute capability 6.0+ for better power management
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        return config.matrixSize * config.matrixSize * sizeof(float);
    }
};

REGISTER_KERNEL(ThermalGradientKernel)