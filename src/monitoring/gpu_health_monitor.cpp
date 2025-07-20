#include "metrics_collector.h"
#include "monitoring_types.h"
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstring>
#include <mutex>

// GPU Health Monitor - collects health metrics via NVML
class GPUHealthMonitor : public MetricsCollector {
public:
    GPUHealthMonitor() : device_(nullptr), deviceIndex_(0) {
        name_ = "GPUHealthMonitor";
        type_ = MetricType::HEALTH;
        collectionInterval_ = std::chrono::milliseconds(100);
    }
    
    ~GPUHealthMonitor() override {
        cleanup();
    }
    
    bool initialize(const CollectorConfig& config) override {
        config_ = config;
        deviceIndex_ = getDeviceId();
        
        // Initialize NVML if not already done
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS && result != NVML_ERROR_ALREADY_INITIALIZED) {
            setLastError("Failed to initialize NVML: " + std::string(nvmlErrorString(result)));
            return false;
        }
        
        // Get device handle
        result = nvmlDeviceGetHandleByIndex(deviceIndex_, &device_);
        if (result != NVML_SUCCESS) {
            setLastError("Failed to get device handle: " + std::string(nvmlErrorString(result)));
            return false;
        }
        
        // Get device name for logging
        char deviceName[NVML_DEVICE_NAME_BUFFER_SIZE];
        result = nvmlDeviceGetName(device_, deviceName, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (result == NVML_SUCCESS) {
            deviceName_ = deviceName;
        } else {
            deviceName_ = "Unknown GPU";
        }
        
        return true;
    }
    
    void cleanup() override {
        // NVML cleanup is handled globally, don't shut it down here
        // as other collectors might still be using it
        device_ = nullptr;
    }
    
    MetricResult collect() override {
        if (!device_) {
            return createFailedResult("Device not initialized");
        }
        
        MetricResult result = createSuccessResult();
        
        // Temperature
        if (!collectTemperature(result)) {
            result.success = false;
        }
        
        // Power
        if (!collectPower(result)) {
            result.success = false;
        }
        
        // Clocks
        if (!collectClocks(result)) {
            result.success = false;
        }
        
        // Fan speed
        if (!collectFanSpeed(result)) {
            // Non-critical, some GPUs don't have fans
        }
        
        // Memory info
        if (!collectMemoryInfo(result)) {
            result.success = false;
        }
        
        // Throttling info
        collectThrottlingInfo(result);
        
        // PCIe info
        collectPCIeInfo(result);
        
        return result;
    }
    
    bool isSupported(int deviceId) const override {
        nvmlDevice_t testDevice;
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(deviceId, &testDevice);
        return result == NVML_SUCCESS;
    }
    
    std::string getName() const override { return name_; }
    MetricType getType() const override { return type_; }
    std::chrono::milliseconds getCollectionInterval() const override { return collectionInterval_; }
    
private:
    bool collectTemperature(MetricResult& result) {
        unsigned int temp;
        
        // GPU temperature
        nvmlReturn_t nvmlResult = nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.temperature"] = static_cast<double>(temp);
        } else {
            result.errorMessage += "Failed to get GPU temperature; ";
            return false;
        }
        
        // Memory temperature (if supported)
        nvmlResult = nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_COUNT, &temp);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.memory_temperature"] = static_cast<double>(temp);
        }
        
        return true;
    }
    
    bool collectPower(MetricResult& result) {
        unsigned int power;
        nvmlReturn_t nvmlResult = nvmlDeviceGetPowerUsage(device_, &power);
        
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.power"] = static_cast<double>(power) / 1000.0;  // Convert mW to W
        } else {
            result.errorMessage += "Failed to get power usage; ";
            return false;
        }
        
        // Also get power limit
        unsigned int powerLimit;
        nvmlResult = nvmlDeviceGetPowerManagementLimit(device_, &powerLimit);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.power_limit"] = static_cast<double>(powerLimit) / 1000.0;
        }
        
        return true;
    }
    
    bool collectClocks(MetricResult& result) {
        unsigned int clock;
        
        // Core clock
        nvmlReturn_t nvmlResult = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_GRAPHICS, &clock);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.core_clock"] = static_cast<double>(clock);
        } else {
            result.errorMessage += "Failed to get core clock; ";
            return false;
        }
        
        // Memory clock
        nvmlResult = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_MEM, &clock);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.memory_clock"] = static_cast<double>(clock);
        } else {
            result.errorMessage += "Failed to get memory clock; ";
            return false;
        }
        
        // SM clock
        nvmlResult = nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.sm_clock"] = static_cast<double>(clock);
        }
        
        return true;
    }
    
    bool collectFanSpeed(MetricResult& result) {
        unsigned int fanSpeed;
        nvmlReturn_t nvmlResult = nvmlDeviceGetFanSpeed(device_, &fanSpeed);
        
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.fan_speed"] = static_cast<double>(fanSpeed);
            return true;
        } else if (nvmlResult == NVML_ERROR_NOT_SUPPORTED) {
            // Device doesn't have a fan (passively cooled or water cooled)
            result.values["gpu.fan_speed"] = 0.0;
            return true;
        } else {
            result.errorMessage += "Failed to get fan speed; ";
            return false;
        }
    }
    
    bool collectMemoryInfo(MetricResult& result) {
        nvmlMemory_t memory;
        nvmlReturn_t nvmlResult = nvmlDeviceGetMemoryInfo(device_, &memory);
        
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.memory_used"] = static_cast<double>(memory.used) / (1024.0 * 1024.0 * 1024.0);  // Convert to GB
            result.values["gpu.memory_free"] = static_cast<double>(memory.free) / (1024.0 * 1024.0 * 1024.0);
            result.values["gpu.memory_total"] = static_cast<double>(memory.total) / (1024.0 * 1024.0 * 1024.0);
            result.values["gpu.memory_utilization"] = (static_cast<double>(memory.used) / memory.total) * 100.0;
            return true;
        } else {
            result.errorMessage += "Failed to get memory info; ";
            return false;
        }
    }
    
    void collectThrottlingInfo(MetricResult& result) {
        unsigned long long throttleReasons;
        nvmlReturn_t nvmlResult = nvmlDeviceGetCurrentClocksThrottleReasons(device_, &throttleReasons);
        
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.throttle_reasons"] = static_cast<double>(throttleReasons);
            
            // Decode specific reasons
            result.values["gpu.throttle_gpu_idle"] = (throttleReasons & nvmlClocksThrottleReasonGpuIdle) ? 1.0 : 0.0;
            result.values["gpu.throttle_sw_power_cap"] = (throttleReasons & nvmlClocksThrottleReasonSwPowerCap) ? 1.0 : 0.0;
            result.values["gpu.throttle_hw_slowdown"] = (throttleReasons & nvmlClocksThrottleReasonHwSlowdown) ? 1.0 : 0.0;
            result.values["gpu.throttle_sw_thermal"] = (throttleReasons & nvmlClocksThrottleReasonSwThermalSlowdown) ? 1.0 : 0.0;
            result.values["gpu.throttle_hw_thermal"] = (throttleReasons & nvmlClocksThrottleReasonHwThermalSlowdown) ? 1.0 : 0.0;
            result.values["gpu.throttle_hw_power_brake"] = (throttleReasons & nvmlClocksThrottleReasonHwPowerBrakeSlowdown) ? 1.0 : 0.0;
        }
        
        // Performance state
        nvmlPstates_t pstate;
        nvmlResult = nvmlDeviceGetPerformanceState(device_, &pstate);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.performance_state"] = static_cast<double>(pstate);
        }
    }
    
    void collectPCIeInfo(MetricResult& result) {
        unsigned int linkGen, linkWidth;
        
        // Current PCIe link generation
        nvmlReturn_t nvmlResult = nvmlDeviceGetCurrPcieLinkGeneration(device_, &linkGen);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.pcie_link_gen"] = static_cast<double>(linkGen);
        }
        
        // Current PCIe link width
        nvmlResult = nvmlDeviceGetCurrPcieLinkWidth(device_, &linkWidth);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.pcie_link_width"] = static_cast<double>(linkWidth);
        }
        
        // PCIe throughput
        unsigned int tx, rx;
        nvmlResult = nvmlDeviceGetPcieThroughput(device_, NVML_PCIE_UTIL_TX_BYTES, &tx);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.pcie_tx_throughput"] = static_cast<double>(tx) / 1024.0;  // KB/s
        }
        
        nvmlResult = nvmlDeviceGetPcieThroughput(device_, NVML_PCIE_UTIL_RX_BYTES, &rx);
        if (nvmlResult == NVML_SUCCESS) {
            result.values["gpu.pcie_rx_throughput"] = static_cast<double>(rx) / 1024.0;  // KB/s
        }
    }
    
private:
    nvmlDevice_t device_;
    unsigned int deviceIndex_;
    std::string deviceName_;
    std::string name_;
    MetricType type_;
    std::chrono::milliseconds collectionInterval_;
};

// Factory function
REGISTER_COLLECTOR(GPUHealthMonitor)

// Global NVML initialization/shutdown management
class NVMLManager {
public:
    static NVMLManager& getInstance() {
        static NVMLManager instance;
        return instance;
    }
    
    bool initialize() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) return true;
        
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS || result == NVML_ERROR_ALREADY_INITIALIZED) {
            initialized_ = true;
            return true;
        }
        return false;
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (initialized_) {
            nvmlShutdown();
            initialized_ = false;
        }
    }
    
private:
    NVMLManager() : initialized_(false) {}
    ~NVMLManager() { shutdown(); }
    
    std::mutex mutex_;
    bool initialized_;
};