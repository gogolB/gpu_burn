#include "metrics_collector.h"
#include "monitoring_types.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cuda_runtime.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#endif

// System Monitor - monitors system-level metrics
class SystemMonitor : public MetricsCollector {
public:
    SystemMonitor() {
        name_ = "SystemMonitor";
        type_ = MetricType::SYSTEM;
        collectionInterval_ = std::chrono::milliseconds(1000);  // 1 second default
    }
    
    ~SystemMonitor() override {
        cleanup();
    }
    
    bool initialize(const CollectorConfig& config) override {
        config_ = config;
        
#ifdef __linux__
        // Get process ID
        pid_ = getpid();
        
        // Get number of CPUs
        numCpus_ = sysconf(_SC_NPROCESSORS_ONLN);
        
        // Get page size for memory calculations
        pageSize_ = sysconf(_SC_PAGE_SIZE);
#endif
        
        return true;
    }
    
    void cleanup() override {
        // Nothing specific to clean up
    }
    
    MetricResult collect() override {
        MetricResult result = createSuccessResult();
        
        // CPU usage
        collectCPUUsage(result);
        
        // Memory usage
        collectMemoryUsage(result);
        
        // PCIe bandwidth (if available)
        collectPCIeBandwidth(result);
        
        // Process-specific metrics
        collectProcessMetrics(result);
        
        // CUDA-specific system metrics
        collectCUDASystemMetrics(result);
        
        return result;
    }
    
    bool isSupported(int deviceId) const override {
        // System monitoring is always supported
        return true;
    }
    
    std::string getName() const override { return name_; }
    MetricType getType() const override { return type_; }
    std::chrono::milliseconds getCollectionInterval() const override { return collectionInterval_; }
    
private:
    void collectCPUUsage(MetricResult& result) {
#ifdef __linux__
        // Read CPU statistics from /proc/stat
        std::ifstream statFile("/proc/stat");
        if (!statFile.is_open()) {
            result.values["system.cpu_usage_percent"] = 0.0;
            return;
        }
        
        std::string line;
        std::getline(statFile, line);
        
        if (line.substr(0, 3) == "cpu") {
            std::istringstream iss(line);
            std::string cpu;
            long user, nice, system, idle, iowait, irq, softirq, steal;
            
            iss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
            
            long total = user + nice + system + idle + iowait + irq + softirq + steal;
            long active = total - idle - iowait;
            
            if (lastTotalCpu_ > 0) {
                long totalDiff = total - lastTotalCpu_;
                long activeDiff = active - lastActiveCpu_;
                
                if (totalDiff > 0) {
                    result.values["system.cpu_usage_percent"] = 
                        (static_cast<double>(activeDiff) / totalDiff) * 100.0;
                }
            }
            
            lastTotalCpu_ = total;
            lastActiveCpu_ = active;
        }
        
        statFile.close();
        
        // Get load average
        double loadavg[3];
        if (getloadavg(loadavg, 3) != -1) {
            result.values["system.load_avg_1min"] = loadavg[0];
            result.values["system.load_avg_5min"] = loadavg[1];
            result.values["system.load_avg_15min"] = loadavg[2];
        }
#else
        result.values["system.cpu_usage_percent"] = 0.0;
#endif
    }
    
    void collectMemoryUsage(MetricResult& result) {
#ifdef __linux__
        struct sysinfo si;
        if (sysinfo(&si) == 0) {
            // System memory
            double totalMem = si.totalram * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
            double freeMem = si.freeram * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
            double usedMem = totalMem - freeMem;
            
            result.values["system.memory_total_gb"] = totalMem;
            result.values["system.memory_used_gb"] = usedMem;
            result.values["system.memory_free_gb"] = freeMem;
            result.values["system.memory_usage_percent"] = (usedMem / totalMem) * 100.0;
            
            // Swap
            double totalSwap = si.totalswap * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
            double freeSwap = si.freeswap * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
            
            result.values["system.swap_total_gb"] = totalSwap;
            result.values["system.swap_used_gb"] = totalSwap - freeSwap;
            result.values["system.swap_free_gb"] = freeSwap;
        }
#else
        result.values["system.memory_used_gb"] = 0.0;
#endif
    }
    
    void collectPCIeBandwidth(MetricResult& result) {
        // This would require reading from PCIe performance counters
        // For now, we'll use a placeholder
        result.values["system.pcie_bandwidth_gbps"] = 0.0;
        
        // Try to get some info from CUDA
        int device;
        if (cudaGetDevice(&device) == cudaSuccess) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
                // Theoretical max PCIe bandwidth
                // Gen3 x16 = 16 GB/s, Gen4 x16 = 32 GB/s
                int pciGen = 3;  // Assume Gen3 by default
                result.values["system.pcie_max_bandwidth_gbps"] = pciGen == 4 ? 32.0 : 16.0;
            }
        }
    }
    
    void collectProcessMetrics(MetricResult& result) {
#ifdef __linux__
        // Process memory usage
        std::string statusPath = "/proc/" + std::to_string(pid_) + "/status";
        std::ifstream statusFile(statusPath);
        
        if (statusFile.is_open()) {
            std::string line;
            while (std::getline(statusFile, line)) {
                if (line.find("VmRSS:") == 0) {
                    long vmRssKb;
                    sscanf(line.c_str(), "VmRSS: %ld kB", &vmRssKb);
                    result.values["system.process_memory_gb"] = vmRssKb / (1024.0 * 1024.0);
                } else if (line.find("VmSize:") == 0) {
                    long vmSizeKb;
                    sscanf(line.c_str(), "VmSize: %ld kB", &vmSizeKb);
                    result.values["system.process_virtual_memory_gb"] = vmSizeKb / (1024.0 * 1024.0);
                } else if (line.find("Threads:") == 0) {
                    int threads;
                    sscanf(line.c_str(), "Threads: %d", &threads);
                    result.values["system.thread_count"] = static_cast<double>(threads);
                }
            }
            statusFile.close();
        }
        
        // Process CPU usage
        std::string statPath = "/proc/" + std::to_string(pid_) + "/stat";
        std::ifstream statFile(statPath);
        
        if (statFile.is_open()) {
            std::string stat;
            std::getline(statFile, stat);
            
            // Parse the stat file (fields are space-separated)
            std::istringstream iss(stat);
            std::string field;
            std::vector<std::string> fields;
            
            while (iss >> field) {
                fields.push_back(field);
            }
            
            if (fields.size() > 14) {
                long utime = std::stol(fields[13]);
                long stime = std::stol(fields[14]);
                long totalTime = utime + stime;
                
                if (lastProcessTime_ > 0) {
                    long timeDiff = totalTime - lastProcessTime_;
                    double cpuUsage = (static_cast<double>(timeDiff) / sysconf(_SC_CLK_TCK)) * 100.0;
                    result.values["system.process_cpu_percent"] = cpuUsage;
                }
                
                lastProcessTime_ = totalTime;
            }
            
            statFile.close();
        }
#else
        result.values["system.process_memory_gb"] = 0.0;
        result.values["system.thread_count"] = 0.0;
#endif
    }
    
    void collectCUDASystemMetrics(MetricResult& result) {
        // Number of CUDA devices
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err == cudaSuccess) {
            result.values["system.cuda_device_count"] = static_cast<double>(deviceCount);
        }
        
        // Current device
        int currentDevice = 0;
        err = cudaGetDevice(&currentDevice);
        if (err == cudaSuccess) {
            result.values["system.cuda_current_device"] = static_cast<double>(currentDevice);
            
            // Get driver version
            int driverVersion = 0;
            cudaDriverGetVersion(&driverVersion);
            result.values["system.cuda_driver_version"] = static_cast<double>(driverVersion);
            
            // Get runtime version
            int runtimeVersion = 0;
            cudaRuntimeGetVersion(&runtimeVersion);
            result.values["system.cuda_runtime_version"] = static_cast<double>(runtimeVersion);
        }
        
        // Check for peer access between devices
        if (deviceCount > 1) {
            int peerAccessCount = 0;
            for (int i = 0; i < deviceCount; i++) {
                for (int j = 0; j < deviceCount; j++) {
                    if (i != j) {
                        int canAccess = 0;
                        cudaDeviceCanAccessPeer(&canAccess, i, j);
                        if (canAccess) {
                            peerAccessCount++;
                        }
                    }
                }
            }
            result.values["system.cuda_peer_access_count"] = static_cast<double>(peerAccessCount);
        }
    }
    
private:
    std::string name_;
    MetricType type_;
    std::chrono::milliseconds collectionInterval_;
    
#ifdef __linux__
    pid_t pid_;
    long numCpus_;
    long pageSize_;
    long lastTotalCpu_ = 0;
    long lastActiveCpu_ = 0;
    long lastProcessTime_ = 0;
#endif
};

// Factory function
REGISTER_COLLECTOR(SystemMonitor)