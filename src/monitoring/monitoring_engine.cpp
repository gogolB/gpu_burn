#include "monitoring_engine.h"
#include "data_store.h"
#include "report_generator.h"
#include "kernel_interface.h"
#include "validation_types.h"
#include <algorithm>
#include <thread>
#include <chrono>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

MonitoringEngine::MonitoringEngine() 
    : stats_{}, startTime_(std::chrono::system_clock::now()) {
}

MonitoringEngine::~MonitoringEngine() {
    shutdown();
}

bool MonitoringEngine::initialize(const MonitoringConfig& config) {
    try {
        config_ = config;
        
        // Initialize data store
        dataStore_ = createDataStore("circular_buffer");
        if (!dataStore_) {
            setLastError("Failed to create data store");
            return false;
        }
        
        RetentionPolicy policy;
        policy.maxAge = config.retentionPeriod;
        policy.maxSamples = config.maxSamplesPerMetric;
        dataStore_->initialize(policy);
        
        // Reset statistics
        stats_ = MonitoringStats{};
        stats_.startTime = std::chrono::system_clock::now();
        startTime_ = stats_.startTime;
        
        return true;
    } catch (const std::exception& e) {
        setLastError(std::string("Initialization failed: ") + e.what());
        return false;
    }
}

void MonitoringEngine::shutdown() {
    // Stop monitoring if running
    stopMonitoring();
    
    // Clear collectors
    {
        std::lock_guard<std::mutex> lock(collectorsMutex_);
        collectors_.clear();
        collectorMap_.clear();
    }
    
    // Clear report generators
    {
        std::lock_guard<std::mutex> lock(reportMutex_);
        reportGenerators_.clear();
    }
}

void MonitoringEngine::registerCollector(std::unique_ptr<MetricsCollector> collector) {
    if (!collector) return;
    
    std::lock_guard<std::mutex> lock(collectorsMutex_);
    
    std::string name = collector->getName();
    collector->setMonitoringEngine(this);
    collector->setDeviceId(config_.enableGPUHealth ? 0 : -1);  // Default to device 0
    
    CollectorConfig collectorConfig;
    collectorConfig.type = collector->getType();
    collectorConfig.collectionInterval = collector->getCollectionInterval();
    collectorConfig.enabled = true;
    
    // Check if type is enabled in config
    MetricType type = collector->getType();
    if (type == MetricType::HEALTH && !config_.enableGPUHealth) {
        collectorConfig.enabled = false;
    } else if (type == MetricType::PERFORMANCE && !config_.enablePerformanceMetrics) {
        collectorConfig.enabled = false;
    } else if (type == MetricType::ERROR && !config_.enableErrorTracking) {
        collectorConfig.enabled = false;
    } else if (type == MetricType::SYSTEM && !config_.enableSystemMetrics) {
        collectorConfig.enabled = false;
    }
    
    collector->initialize(collectorConfig);
    collector->enable(collectorConfig.enabled);
    
    collectorMap_[name] = collector.get();
    collectors_.push_back(std::move(collector));
}

void MonitoringEngine::unregisterCollector(const std::string& name) {
    std::lock_guard<std::mutex> lock(collectorsMutex_);
    
    auto it = collectorMap_.find(name);
    if (it != collectorMap_.end()) {
        collectorMap_.erase(it);
        
        // Find and remove from vector
        collectors_.erase(
            std::remove_if(collectors_.begin(), collectors_.end(),
                [&name](const std::unique_ptr<MetricsCollector>& c) {
                    return c->getName() == name;
                }),
            collectors_.end()
        );
    }
}

MetricsCollector* MonitoringEngine::getCollector(const std::string& name) {
    std::lock_guard<std::mutex> lock(collectorsMutex_);
    auto it = collectorMap_.find(name);
    return (it != collectorMap_.end()) ? it->second : nullptr;
}

std::vector<std::string> MonitoringEngine::getCollectorNames() const {
    std::lock_guard<std::mutex> lock(collectorsMutex_);
    std::vector<std::string> names;
    names.reserve(collectorMap_.size());
    for (const auto& pair : collectorMap_) {
        names.push_back(pair.first);
    }
    return names;
}

void MonitoringEngine::startMonitoring() {
    if (isMonitoring_.exchange(true)) {
        return;  // Already monitoring
    }
    
    shouldStop_ = false;
    monitoringThread_ = std::thread(&MonitoringEngine::monitoringThreadFunc, this);
}

void MonitoringEngine::stopMonitoring() {
    if (!isMonitoring_.exchange(false)) {
        return;  // Not monitoring
    }
    
    shouldStop_ = true;
    monitoringCV_.notify_all();
    
    if (monitoringThread_.joinable()) {
        monitoringThread_.join();
    }
}

MonitoringSnapshot MonitoringEngine::captureSnapshot() {
    return captureSnapshot(MetricType::ALL);
}

MonitoringSnapshot MonitoringEngine::captureSnapshot(MetricType types) {
    MonitoringSnapshot snapshot;
    
    // Get latest values from data store
    if (dataStore_) {
        // Health metrics
        if (isMetricTypeEnabled(types, MetricType::HEALTH)) {
            snapshot.health.temperature = dataStore_->getLatestValue("gpu.temperature");
            snapshot.health.power = dataStore_->getLatestValue("gpu.power");
            snapshot.health.coreClockMHz = dataStore_->getLatestValue("gpu.core_clock");
            snapshot.health.memoryClockMHz = dataStore_->getLatestValue("gpu.memory_clock");
            snapshot.health.fanSpeedPercent = dataStore_->getLatestValue("gpu.fan_speed");
        }
        
        // Performance metrics
        if (isMetricTypeEnabled(types, MetricType::PERFORMANCE)) {
            snapshot.performance.currentGFLOPS = dataStore_->getLatestValue("perf.gflops");
            snapshot.performance.avgGFLOPS = dataStore_->getLatestValue("perf.avg_gflops");
            snapshot.performance.peakGFLOPS = dataStore_->getLatestValue("perf.peak_gflops");
            snapshot.performance.bandwidthGBps = dataStore_->getLatestValue("perf.bandwidth");
            snapshot.performance.smEfficiency = dataStore_->getLatestValue("perf.sm_efficiency");
        }
        
        // Error metrics
        if (isMetricTypeEnabled(types, MetricType::ERROR)) {
            snapshot.errors.totalErrors = static_cast<size_t>(dataStore_->getLatestValue("error.total"));
            snapshot.errors.recentErrors = static_cast<size_t>(dataStore_->getLatestValue("error.recent"));
            
            // TODO: Populate error types from collector
        }
    }
    
    // System status
    {
        std::lock_guard<std::mutex> lock(alertsMutex_);
        snapshot.status.activeAlerts = activeAlerts_;
        
        // Check throttling from health metrics
        double temp = snapshot.health.temperature;
        if (temp > config_.thresholds.maxTemperature) {
            snapshot.status.isThrottling = true;
            snapshot.status.throttleReasons.push_back("High temperature");
        }
    }
    
    return snapshot;
}

TimeSeriesData MonitoringEngine::getHistoricalData(MetricType type, const TimeRange& range) {
    if (!dataStore_) {
        return TimeSeriesData{};
    }
    
    // Get all metrics of the specified type
    auto allMetrics = dataStore_->getMetricNames(type);
    if (allMetrics.empty()) {
        return TimeSeriesData{};
    }
    
    // For now, return data for the first metric of this type
    return dataStore_->query(allMetrics[0], range);
}

TimeSeriesData MonitoringEngine::getHistoricalData(const std::string& metricName, const TimeRange& range) {
    if (!dataStore_) {
        return TimeSeriesData{};
    }
    
    return dataStore_->query(metricName, range);
}

MonitoringStats MonitoringEngine::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    MonitoringStats currentStats = stats_;
    currentStats.uptime = std::chrono::system_clock::now() - startTime_;
    currentStats.collectionSuccessRate = (stats_.totalSamplesCollected > 0) 
        ? (double)(stats_.totalSamplesCollected - stats_.failedCollections) / stats_.totalSamplesCollected 
        : 0.0;
    return currentStats;
}

void MonitoringEngine::recordKernelExecution(const KernelResult& result) {
    kernelExecutionCount_++;
    
    // Use atomic operation for double
    double oldValue = totalKernelTimeMs_.load();
    while (!totalKernelTimeMs_.compare_exchange_weak(oldValue, oldValue + result.executionTimeMs)) {
        // Loop will retry if another thread modified the value
    }
    
    if (dataStore_) {
        // Record performance metrics
        dataStore_->insert("kernel.execution_time", result.executionTimeMs);
        dataStore_->insert("kernel.gflops", result.gflops);
        dataStore_->insert("kernel.bandwidth", static_cast<double>(result.memoryBandwidthGBps));
        
        // Record validation metrics if performed
        if (result.validationPerformed) {
            validationCount_++;
            sdcCount_ += result.sdcDetectedCount;
            
            dataStore_->insert("validation.sdc_detected", static_cast<double>(result.sdcDetectedCount));
            dataStore_->insert("validation.sdc_corrected", static_cast<double>(result.sdcCorrectedCount));
            dataStore_->insert("validation.overhead_ms", result.validationOverheadMs);
        }
    }
}

void MonitoringEngine::recordValidationResult(const ValidationResult& result) {
    if (!dataStore_) return;
    
    std::string prefix = "validation." + getValidationTypeName(result.method) + ".";
    dataStore_->insert(prefix + "passed", result.passed ? 1.0 : 0.0);
    dataStore_->insert(prefix + "confidence", result.confidence);
    dataStore_->insert(prefix + "corrupted_elements", static_cast<double>(result.corruptedElements));
    dataStore_->insert(prefix + "time_ms", result.validationTimeMs);
}

void MonitoringEngine::registerAlert(const Alert& alert) {
    std::lock_guard<std::mutex> lock(alertsMutex_);
    alerts_.push_back(alert);
}

void MonitoringEngine::unregisterAlert(const std::string& alertId) {
    std::lock_guard<std::mutex> lock(alertsMutex_);
    alerts_.erase(
        std::remove_if(alerts_.begin(), alerts_.end(),
            [&alertId](const Alert& a) { return a.id == alertId; }),
        alerts_.end()
    );
}

std::vector<std::string> MonitoringEngine::getActiveAlerts() const {
    std::lock_guard<std::mutex> lock(alertsMutex_);
    return activeAlerts_;
}

void MonitoringEngine::registerReportGenerator(std::unique_ptr<ReportGenerator> generator) {
    if (!generator) return;
    
    std::lock_guard<std::mutex> lock(reportMutex_);
    generator->setMonitoringEngine(this);
    reportGenerators_[generator->getFormat()] = std::move(generator);
}

void MonitoringEngine::generateReport(const std::string& format) {
    std::lock_guard<std::mutex> lock(reportMutex_);
    
    auto it = reportGenerators_.find(format);
    if (it != reportGenerators_.end()) {
        auto snapshot = captureSnapshot();
        auto historical = getHistoricalData(MetricType::ALL, TimeRange::lastMinutes(10));
        it->second->generate(snapshot, historical);
    }
}

void MonitoringEngine::updateConfig(const MonitoringConfig& config) {
    config_ = config;
    
    // Update retention policy
    if (dataStore_) {
        RetentionPolicy policy;
        policy.maxAge = config.retentionPeriod;
        policy.maxSamples = config.maxSamplesPerMetric;
        dataStore_->setRetentionPolicy(policy);
    }
    
    // Update collector enablement
    std::lock_guard<std::mutex> lock(collectorsMutex_);
    for (auto& collector : collectors_) {
        MetricType type = collector->getType();
        bool shouldEnable = true;
        
        if (type == MetricType::HEALTH) shouldEnable = config.enableGPUHealth;
        else if (type == MetricType::PERFORMANCE) shouldEnable = config.enablePerformanceMetrics;
        else if (type == MetricType::ERROR) shouldEnable = config.enableErrorTracking;
        else if (type == MetricType::SYSTEM) shouldEnable = config.enableSystemMetrics;
        
        collector->enable(shouldEnable);
    }
}

void MonitoringEngine::monitoringThreadFunc() {
    setThreadPriority(config_.monitoringThreadPriority);
    if (config_.monitoringThreadAffinity != -1) {
        setThreadAffinity(config_.monitoringThreadAffinity);
    }
    
    while (!shouldStop_.load()) {
        auto startTime = std::chrono::steady_clock::now();
        
        // Collect from all registered collectors
        {
            std::lock_guard<std::mutex> lock(collectorsMutex_);
            for (auto& collector : collectors_) {
                if (!collector->isEnabled()) continue;
                
                if (shouldCollect(collector.get(), startTime)) {
                    auto result = collector->collect();
                    processCollectorResult(collector.get(), result);
                }
            }
        }
        
        // Check for alerts
        {
            std::lock_guard<std::mutex> lock(alertsMutex_);
            activeAlerts_.clear();
            
            for (auto& alert : alerts_) {
                auto value = dataStore_->getLatestValue(alert.metricName);
                checkAlert(alert, MetricResult{});  // TODO: Pass actual result
            }
        }
        
        // Sleep until next collection interval
        auto elapsed = std::chrono::steady_clock::now() - startTime;
        auto sleepTime = config_.globalSamplingInterval - 
                        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        
        if (sleepTime.count() > 0) {
            std::unique_lock<std::mutex> lock(monitoringMutex_);
            monitoringCV_.wait_for(lock, sleepTime, [this] { return shouldStop_.load(); });
        }
    }
}

bool MonitoringEngine::shouldCollect(MetricsCollector* collector, 
                                   const std::chrono::steady_clock::time_point& currentTime) {
    auto it = lastCollectionTimes_.find(collector->getName());
    if (it == lastCollectionTimes_.end()) {
        lastCollectionTimes_[collector->getName()] = currentTime;
        return true;
    }
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - it->second);
    auto interval = collector->getCollectionInterval();
    
    // Check if custom interval is set for this metric type
    auto customIt = config_.customIntervals.find(collector->getType());
    if (customIt != config_.customIntervals.end()) {
        interval = customIt->second;
    }
    
    if (elapsed >= interval) {
        it->second = currentTime;
        return true;
    }
    
    return false;
}

void MonitoringEngine::processCollectorResult(MetricsCollector* /* collector */, const MetricResult& result) {
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.totalSamplesCollected++;
        stats_.samplesByType[result.type]++;
        
        if (!result.success) {
            stats_.failedCollections++;
        }
    }
    
    if (result.success && dataStore_) {
        // Insert all values from the result
        for (const auto& [name, value] : result.values) {
            dataStore_->insert(name, value);
        }
        
        // Check thresholds
        checkThresholds(result.type, result);
    }
}

void MonitoringEngine::checkThresholds(MetricType type, const MetricResult& result) {
    if (type == MetricType::HEALTH) {
        auto tempIt = result.values.find("gpu.temperature");
        if (tempIt != result.values.end() && tempIt->second > config_.thresholds.maxTemperature) {
            std::lock_guard<std::mutex> lock(alertsMutex_);
            activeAlerts_.push_back("High GPU temperature: " + std::to_string(tempIt->second) + "°C");
        }
        
        auto powerIt = result.values.find("gpu.power");
        if (powerIt != result.values.end() && powerIt->second > config_.thresholds.maxPower) {
            std::lock_guard<std::mutex> lock(alertsMutex_);
            activeAlerts_.push_back("High power consumption: " + std::to_string(powerIt->second) + "W");
        }
    }
}

void MonitoringEngine::checkAlert(const Alert& alert, const MetricResult& /* result */) {
    auto value = dataStore_->getLatestValue(alert.metricName);
    
    if (alert.condition && alert.condition(value)) {
        auto now = std::chrono::system_clock::now();
        auto timeSinceLastTrigger = now - alert.lastTriggered;
        
        if (timeSinceLastTrigger >= alert.cooldownPeriod) {
            activeAlerts_.push_back(alert.name + ": " + alert.description);
            const_cast<Alert&>(alert).lastTriggered = now;
            
            if (alertCallback_) {
                alertCallback_(alert, value);
            }
        }
    }
}

void MonitoringEngine::setThreadPriority(ThreadPriority priority) {
#ifdef __linux__
    int policy = SCHED_OTHER;
    struct sched_param param;
    param.sched_priority = 0;
    
    switch (priority) {
        case ThreadPriority::REALTIME:
            policy = SCHED_FIFO;
            param.sched_priority = 10;
            break;
        case ThreadPriority::HIGH:
            policy = SCHED_OTHER;
            // Use nice value instead
            if (nice(-5) == -1) {
                // nice() can fail, but we'll continue anyway
            }
            break;
        case ThreadPriority::LOW:
            if (nice(5) == -1) {
                // nice() can fail, but we'll continue anyway
            }
            break;
        default:
            break;
    }
    
    pthread_setschedparam(pthread_self(), policy, &param);
#endif
}

void MonitoringEngine::setThreadAffinity(int cpuCore) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuCore, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}