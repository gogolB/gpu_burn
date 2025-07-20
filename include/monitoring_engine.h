#ifndef MONITORING_ENGINE_H
#define MONITORING_ENGINE_H

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "monitoring_types.h"
#include "metrics_collector.h"

// Forward declarations
class DataStore;
class ReportGenerator;
struct KernelResult;
struct ValidationResult;

// Main monitoring orchestrator
class MonitoringEngine {
public:
    MonitoringEngine();
    ~MonitoringEngine();
    
    // Lifecycle management
    bool initialize(const MonitoringConfig& config);
    void shutdown();
    
    // Collector registration
    void registerCollector(std::unique_ptr<MetricsCollector> collector);
    void unregisterCollector(const std::string& name);
    MetricsCollector* getCollector(const std::string& name);
    std::vector<std::string> getCollectorNames() const;
    
    // Main monitoring interface
    void startMonitoring();
    void stopMonitoring();
    bool isMonitoring() const { return isMonitoring_.load(); }
    
    // Data capture
    MonitoringSnapshot captureSnapshot();
    MonitoringSnapshot captureSnapshot(MetricType types);
    
    // Data access
    TimeSeriesData getHistoricalData(MetricType type, const TimeRange& range);
    TimeSeriesData getHistoricalData(const std::string& metricName, const TimeRange& range);
    MonitoringStats getStats() const;
    
    // Integration with kernel execution
    void recordKernelExecution(const KernelResult& result);
    void recordValidationResult(const ValidationResult& result);
    
    // Alert management
    void registerAlert(const Alert& alert);
    void unregisterAlert(const std::string& alertId);
    std::vector<std::string> getActiveAlerts() const;
    
    // Report generation
    void registerReportGenerator(std::unique_ptr<ReportGenerator> generator);
    void generateReport(const std::string& format = "console");
    
    // Data store access
    DataStore* getDataStore() { return dataStore_.get(); }
    const DataStore* getDataStore() const { return dataStore_.get(); }
    
    // Configuration
    const MonitoringConfig& getConfig() const { return config_; }
    void updateConfig(const MonitoringConfig& config);
    
    // Error handling
    std::string getLastError() const { return lastError_; }
    size_t getErrorCount() const { return errorCount_; }
    
    // Callbacks for events
    using AlertCallback = std::function<void(const Alert&, double value)>;
    void setAlertCallback(AlertCallback callback) { alertCallback_ = callback; }
    
private:
    // Background monitoring thread
    void monitoringThreadFunc();
    
    // Collection helpers
    bool shouldCollect(MetricsCollector* collector, 
                      const std::chrono::steady_clock::time_point& currentTime);
    void processCollectorResult(MetricsCollector* collector, const MetricResult& result);
    
    // Alert checking
    void checkThresholds(MetricType type, const MetricResult& result);
    void checkAlert(const Alert& alert, const MetricResult& result);
    
    // Thread management
    void setThreadPriority(ThreadPriority priority);
    void setThreadAffinity(int cpuCore);
    
    // Error handling
    void setLastError(const std::string& error) {
        std::lock_guard<std::mutex> lock(errorMutex_);
        lastError_ = error;
        errorCount_++;
    }
    
private:
    // Configuration
    MonitoringConfig config_;
    
    // Collectors
    std::vector<std::unique_ptr<MetricsCollector>> collectors_;
    std::unordered_map<std::string, MetricsCollector*> collectorMap_;
    mutable std::mutex collectorsMutex_;
    
    // Data storage
    std::unique_ptr<DataStore> dataStore_;
    
    // Background thread
    std::atomic<bool> isMonitoring_{false};
    std::atomic<bool> shouldStop_{false};
    std::thread monitoringThread_;
    std::condition_variable monitoringCV_;
    std::mutex monitoringMutex_;
    
    // Last collection times
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> lastCollectionTimes_;
    
    // Statistics
    mutable std::mutex statsMutex_;
    MonitoringStats stats_;
    std::chrono::system_clock::time_point startTime_;
    
    // Alerts
    std::vector<Alert> alerts_;
    std::vector<std::string> activeAlerts_;
    mutable std::mutex alertsMutex_;
    AlertCallback alertCallback_;
    
    // Report generators
    std::unordered_map<std::string, std::unique_ptr<ReportGenerator>> reportGenerators_;
    mutable std::mutex reportMutex_;
    
    // Error tracking
    mutable std::mutex errorMutex_;
    std::string lastError_;
    std::atomic<size_t> errorCount_{0};
    
    // Kernel/validation integration data
    std::atomic<size_t> kernelExecutionCount_{0};
    std::atomic<double> totalKernelTimeMs_{0.0};
    std::atomic<size_t> validationCount_{0};
    std::atomic<size_t> sdcCount_{0};
};

#endif // MONITORING_ENGINE_H