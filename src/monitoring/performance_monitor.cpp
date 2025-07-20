#include "metrics_collector.h"
#include "monitoring_types.h"
#include "kernel_interface.h"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <mutex>
#include <deque>

// Performance Monitor - collects kernel execution and performance metrics
class PerformanceMonitor : public MetricsCollector {
public:
    PerformanceMonitor() : kernelCount_(0), totalTimeMs_(0.0), totalGflops_(0.0) {
        name_ = "PerformanceMonitor";
        type_ = MetricType::PERFORMANCE;
        collectionInterval_ = std::chrono::milliseconds(100);
    }
    
    ~PerformanceMonitor() override {
        cleanup();
    }
    
    bool initialize(const CollectorConfig& config) override {
        config_ = config;
        
        // Create CUDA events for timing
        cudaError_t err = cudaEventCreate(&startEvent_);
        if (err != cudaSuccess) {
            setLastError("Failed to create start event: " + std::string(cudaGetErrorString(err)));
            return false;
        }
        
        err = cudaEventCreate(&stopEvent_);
        if (err != cudaSuccess) {
            cudaEventDestroy(startEvent_);
            setLastError("Failed to create stop event: " + std::string(cudaGetErrorString(err)));
            return false;
        }
        
        // Initialize performance history
        maxHistorySize_ = 1000;  // Keep last 1000 kernel executions
        
        return true;
    }
    
    void cleanup() override {
        if (startEvent_) {
            cudaEventDestroy(startEvent_);
            startEvent_ = nullptr;
        }
        if (stopEvent_) {
            cudaEventDestroy(stopEvent_);
            stopEvent_ = nullptr;
        }
    }
    
    MetricResult collect() override {
        MetricResult result = createSuccessResult();
        
        std::lock_guard<std::mutex> lock(historyMutex_);
        
        // Calculate current metrics from history
        if (kernelHistory_.empty()) {
            // No kernel executions yet
            result.values["perf.gflops"] = 0.0;
            result.values["perf.avg_gflops"] = 0.0;
            result.values["perf.peak_gflops"] = 0.0;
            result.values["perf.bandwidth"] = 0.0;
            result.values["perf.sm_efficiency"] = 0.0;
            result.values["perf.kernel_count"] = 0.0;
            result.values["perf.avg_kernel_time"] = 0.0;
            result.values["perf.throughput"] = 0.0;
        } else {
            // Calculate metrics from recent history
            double totalGflops = 0.0;
            double totalBandwidth = 0.0;
            double totalEfficiency = 0.0;
            double totalTime = 0.0;
            double peakGflops = 0.0;
            int count = 0;
            
            // Process recent kernel executions
            auto now = std::chrono::steady_clock::now();
            for (const auto& kernel : kernelHistory_) {
                auto age = std::chrono::duration_cast<std::chrono::seconds>(now - kernel.timestamp);
                if (age.count() < 60) {  // Only consider last minute
                    totalGflops += kernel.gflops;
                    totalBandwidth += kernel.bandwidthGBps;
                    totalEfficiency += kernel.smEfficiency;
                    totalTime += kernel.executionTimeMs;
                    peakGflops = std::max(peakGflops, kernel.gflops);
                    count++;
                }
            }
            
            if (count > 0) {
                result.values["perf.gflops"] = kernelHistory_.back().gflops;  // Current
                result.values["perf.avg_gflops"] = totalGflops / count;
                result.values["perf.peak_gflops"] = peakGflops;
                result.values["perf.bandwidth"] = totalBandwidth / count;
                result.values["perf.sm_efficiency"] = totalEfficiency / count;
                result.values["perf.kernel_count"] = static_cast<double>(count);
                result.values["perf.avg_kernel_time"] = totalTime / count;
                result.values["perf.throughput"] = count / 60.0;  // Kernels per second
            }
        }
        
        // Collect current GPU utilization
        collectGPUUtilization(result);
        
        // Memory performance
        collectMemoryMetrics(result);
        
        return result;
    }
    
    bool isSupported(int deviceId) const override {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
        return err == cudaSuccess;
    }
    
    std::string getName() const override { return name_; }
    MetricType getType() const override { return type_; }
    std::chrono::milliseconds getCollectionInterval() const override { return collectionInterval_; }
    
    // Record kernel execution for performance tracking
    void recordKernelExecution(const KernelResult& result) {
        std::lock_guard<std::mutex> lock(historyMutex_);
        
        KernelPerformance perf;
        perf.timestamp = std::chrono::steady_clock::now();
        perf.executionTimeMs = result.executionTimeMs;
        perf.gflops = result.gflops;
        perf.bandwidthGBps = static_cast<double>(result.memoryBandwidthGBps);
        perf.smEfficiency = calculateSMEfficiency();
        perf.occupancy = calculateOccupancy();
        
        kernelHistory_.push_back(perf);
        
        // Maintain history size
        while (kernelHistory_.size() > maxHistorySize_) {
            kernelHistory_.pop_front();
        }
        
        // Update totals
        kernelCount_++;
        
        // Atomic operations for doubles
        double oldTime = totalTimeMs_.load();
        while (!totalTimeMs_.compare_exchange_weak(oldTime, oldTime + result.executionTimeMs)) {
            // Loop will retry
        }
        
        double oldGflops = totalGflops_.load();
        while (!totalGflops_.compare_exchange_weak(oldGflops, oldGflops + result.gflops)) {
            // Loop will retry
        }
    }
    
    // Start timing for a kernel
    void startTiming() {
        cudaEventRecord(startEvent_);
    }
    
    // Stop timing and get elapsed time
    float stopTiming() {
        cudaEventRecord(stopEvent_);
        cudaEventSynchronize(stopEvent_);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, startEvent_, stopEvent_);
        return milliseconds;
    }
    
private:
    struct KernelPerformance {
        std::chrono::steady_clock::time_point timestamp;
        double executionTimeMs;
        double gflops;
        double bandwidthGBps;
        double smEfficiency;
        double occupancy;
    };
    
    void collectGPUUtilization(MetricResult& result) {
        // Get current device
        int device;
        cudaGetDevice(&device);
        
        // Get device properties
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            return;
        }
        
        // Calculate theoretical peak performance
        // Peak GFLOPS = (cores per SM) * (SMs) * (clock rate in GHz) * 2 (FMA)
        double peakGflops = prop.multiProcessorCount * 
                           getCoresPerSM(prop.major, prop.minor) * 
                           (prop.clockRate / 1000000.0) * 2.0;
        
        result.values["perf.theoretical_peak_gflops"] = peakGflops;
        
        // Get memory info for bandwidth calculation
        size_t freeMem, totalMem;
        err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err == cudaSuccess) {
            result.values["perf.memory_free_gb"] = static_cast<double>(freeMem) / (1024.0 * 1024.0 * 1024.0);
            result.values["perf.memory_used_gb"] = static_cast<double>(totalMem - freeMem) / (1024.0 * 1024.0 * 1024.0);
        }
    }
    
    void collectMemoryMetrics(MetricResult& result) {
        // This would ideally use CUPTI or nvprof APIs for detailed metrics
        // For now, we'll estimate based on kernel history
        
        if (!kernelHistory_.empty()) {
            double totalBandwidth = 0.0;
            int count = 0;
            
            for (const auto& kernel : kernelHistory_) {
                if (kernel.bandwidthGBps > 0) {
                    totalBandwidth += kernel.bandwidthGBps;
                    count++;
                }
            }
            
            if (count > 0) {
                result.values["perf.avg_memory_bandwidth"] = totalBandwidth / count;
            }
        }
        
        // Get device memory bandwidth
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err == cudaSuccess) {
            // Memory bandwidth in GB/s = (memory clock in MHz * bus width in bits * 2) / 8 / 1000
            double peakBandwidth = (prop.memoryClockRate / 1000.0) * (prop.memoryBusWidth / 8.0) * 2.0 / 1000.0;
            result.values["perf.peak_memory_bandwidth"] = peakBandwidth;
        }
    }
    
    double calculateSMEfficiency() {
        // This is a simplified calculation
        // Real implementation would use CUPTI metrics
        return 85.0;  // Placeholder - assume 85% efficiency
    }
    
    double calculateOccupancy() {
        // This is a simplified calculation
        // Real implementation would use cudaOccupancyMaxActiveBlocksPerMultiprocessor
        return 75.0;  // Placeholder - assume 75% occupancy
    }
    
    int getCoresPerSM(int major, int minor) {
        // CUDA cores per SM for different architectures
        switch (major) {
            case 2: return 32;   // Fermi
            case 3: return 192;  // Kepler
            case 5: return 128;  // Maxwell
            case 6: return 64;   // Pascal (FP32)
            case 7: return 64;   // Volta/Turing (FP32)
            case 8: 
                if (minor == 0) return 64;   // Ampere A100 (FP32)
                else return 128;              // Ampere GA10x (FP32)
            case 9: return 128;  // Hopper (FP32)
            default: return 64;  // Default assumption
        }
    }
    
private:
    cudaEvent_t startEvent_ = nullptr;
    cudaEvent_t stopEvent_ = nullptr;
    
    // Performance history
    std::deque<KernelPerformance> kernelHistory_;
    std::mutex historyMutex_;
    size_t maxHistorySize_;
    
    // Aggregate statistics
    std::atomic<size_t> kernelCount_;
    std::atomic<double> totalTimeMs_;
    std::atomic<double> totalGflops_;
    
    std::string name_;
    MetricType type_;
    std::chrono::milliseconds collectionInterval_;
};

// Factory function
REGISTER_COLLECTOR(PerformanceMonitor)