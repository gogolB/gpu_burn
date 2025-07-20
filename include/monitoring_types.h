#ifndef MONITORING_TYPES_H
#define MONITORING_TYPES_H

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <cmath>
#include <functional>

// Metric types
enum class MetricType {
    PERFORMANCE = 1 << 0,
    HEALTH = 1 << 1,
    ERROR = 1 << 2,
    SYSTEM = 1 << 3,
    ALL = PERFORMANCE | HEALTH | ERROR | SYSTEM
};

// Thread priority levels
enum class ThreadPriority {
    LOW,
    NORMAL,
    HIGH,
    REALTIME
};

// Aggregation types for data queries
enum class AggregationType {
    MIN,
    MAX,
    AVERAGE,
    SUM,
    COUNT,
    PERCENTILE_50,
    PERCENTILE_95,
    PERCENTILE_99
};

// Time-series data point
template<typename T>
struct TimeSeriesDataPoint {
    std::chrono::system_clock::time_point timestamp;
    T value;
    
    TimeSeriesDataPoint() = default;
    TimeSeriesDataPoint(const T& val) 
        : timestamp(std::chrono::system_clock::now()), value(val) {}
};

// Time-series data container
struct TimeSeriesData {
    MetricType type;
    std::string metricName;
    std::vector<TimeSeriesDataPoint<double>> dataPoints;
    
    // Statistical summary
    struct Stats {
        double min;
        double max;
        double average;
        double stddev;
        size_t count;
    } stats;
    
    void calculateStats() {
        if (dataPoints.empty()) {
            stats = {0, 0, 0, 0, 0};
            return;
        }
        
        double sum = 0, sum2 = 0;
        stats.min = dataPoints[0].value;
        stats.max = dataPoints[0].value;
        
        for (const auto& dp : dataPoints) {
            sum += dp.value;
            sum2 += dp.value * dp.value;
            stats.min = std::min(stats.min, dp.value);
            stats.max = std::max(stats.max, dp.value);
        }
        
        stats.count = dataPoints.size();
        stats.average = sum / stats.count;
        stats.stddev = std::sqrt(sum2 / stats.count - stats.average * stats.average);
    }
};

// Time range for queries
struct TimeRange {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    
    TimeRange() = default;
    TimeRange(std::chrono::system_clock::time_point s, std::chrono::system_clock::time_point e)
        : start(s), end(e) {}
    
    // Helper to create a range for the last N minutes
    static TimeRange lastMinutes(int minutes) {
        auto now = std::chrono::system_clock::now();
        auto start = now - std::chrono::minutes(minutes);
        return TimeRange(start, now);
    }
};

// Monitoring configuration
struct MonitoringConfig {
    // Sampling configuration
    std::chrono::milliseconds globalSamplingInterval{100};  // 100ms default
    std::unordered_map<MetricType, std::chrono::milliseconds> customIntervals;
    
    // Data retention
    std::chrono::hours retentionPeriod{1};  // Keep 1 hour of data
    size_t maxSamplesPerMetric{36000};      // ~10 samples/sec for 1 hour
    
    // Feature flags
    bool enableGPUHealth{true};
    bool enablePerformanceMetrics{true};
    bool enableErrorTracking{true};
    bool enableSystemMetrics{false};
    
    // Background thread configuration
    ThreadPriority monitoringThreadPriority{ThreadPriority::LOW};
    int monitoringThreadAffinity{-1};  // CPU core affinity (-1 = no affinity)
    
    // Alerting thresholds
    struct Thresholds {
        double maxTemperature{85.0};      // Celsius
        double maxPower{300.0};           // Watts
        double minClockSpeed{1000.0};     // MHz
        size_t maxErrorsPerMinute{10};
    } thresholds;
};

// Collector configuration
struct CollectorConfig {
    MetricType type;
    std::chrono::milliseconds collectionInterval;
    bool enabled;
    std::unordered_map<std::string, std::string> parameters;
};

// Metric result from a collector
struct MetricResult {
    bool success;
    MetricType type;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, double> values;
    std::string errorMessage;
    
    MetricResult() : success(false), type(MetricType::SYSTEM), 
                     timestamp(std::chrono::system_clock::now()) {}
};

// GPU health metrics
struct HealthMetrics {
    double temperatureGPU;      // Celsius
    double temperatureMemory;   // Celsius
    double powerDraw;          // Watts
    double coreClockMHz;
    double memoryClockMHz;
    double fanSpeedPercent;
    double voltage;            // Volts
    uint32_t throttleReasons;  // Bit mask of throttle reasons
};

// Performance metrics
struct PerformanceMetrics {
    double currentGFLOPS;
    double avgGFLOPS;
    double peakGFLOPS;
    double bandwidthGBps;
    double smEfficiency;      // SM utilization percentage
    double occupancy;         // Achieved occupancy
    size_t kernelCount;       // Number of kernels executed
    double avgKernelTimeMs;   // Average kernel execution time
};

// Error tracking metrics
struct ErrorMetrics {
    size_t totalErrors;
    size_t recentErrors;      // Errors in the last minute
    std::unordered_map<std::string, size_t> errorsByType;
    double errorRate;         // Errors per second
    std::vector<std::pair<std::chrono::system_clock::time_point, std::string>> recentErrorLog;
};

// System metrics
struct SystemMetrics {
    double pcieBandwidthGBps;
    double systemMemoryUsageGB;
    double cpuUsagePercent;
    double processMemoryUsageGB;
    size_t threadCount;
};

// Monitoring snapshot - comprehensive view at a point in time
struct MonitoringSnapshot {
    std::chrono::system_clock::time_point timestamp;
    
    // Current values
    struct {
        double temperature;
        double power;
        double coreClockMHz;
        double memoryClockMHz;
        double fanSpeedPercent;
    } health;
    
    // Performance metrics
    struct {
        double currentGFLOPS;
        double avgGFLOPS;
        double peakGFLOPS;
        double bandwidthGBps;
        double smEfficiency;
    } performance;
    
    // Error statistics
    struct {
        size_t totalErrors;
        size_t recentErrors;  // Last minute
        std::unordered_map<std::string, size_t> errorsByType;
    } errors;
    
    // System status
    struct {
        bool isThrottling;
        std::vector<std::string> throttleReasons;
        std::vector<std::string> activeAlerts;
    } status;
    
    MonitoringSnapshot() : timestamp(std::chrono::system_clock::now()) {
        // Initialize with zeros
        health = {0, 0, 0, 0, 0};
        performance = {0, 0, 0, 0, 0};
        errors.totalErrors = 0;
        errors.recentErrors = 0;
        status.isThrottling = false;
    }
};

// Monitoring statistics
struct MonitoringStats {
    size_t totalSamplesCollected;
    size_t failedCollections;
    double collectionSuccessRate;
    std::chrono::system_clock::time_point startTime;
    std::chrono::duration<double> uptime;
    std::unordered_map<MetricType, size_t> samplesByType;
};

// Retention policy for data storage
struct RetentionPolicy {
    std::chrono::hours maxAge{1};
    size_t maxSamples{36000};
    bool compressOldData{false};
    double compressionRatio{0.1};  // Keep 10% of samples when compressing
};

// Alert definition
struct Alert {
    std::string id;
    std::string name;
    std::string description;
    MetricType metricType;
    std::string metricName;
    double threshold;
    std::function<bool(double)> condition;
    std::chrono::seconds cooldownPeriod{60};
    std::chrono::system_clock::time_point lastTriggered;
};

// Helper functions for metric types

// Get metric type name as string
inline std::string getMetricTypeName(MetricType type) {
    switch (type) {
        case MetricType::PERFORMANCE: return "Performance";
        case MetricType::HEALTH: return "Health";
        case MetricType::ERROR: return "Error";
        case MetricType::SYSTEM: return "System";
        case MetricType::ALL: return "All";
        default: return "Unknown";
    }
}

// Check if metric type is enabled
inline bool isMetricTypeEnabled(MetricType enabled, MetricType check) {
    return (static_cast<int>(enabled) & static_cast<int>(check)) != 0;
}

// Combine metric types
inline MetricType operator|(MetricType a, MetricType b) {
    return static_cast<MetricType>(static_cast<int>(a) | static_cast<int>(b));
}

inline MetricType operator&(MetricType a, MetricType b) {
    return static_cast<MetricType>(static_cast<int>(a) & static_cast<int>(b));
}

// Convert thread priority to system value (platform-specific implementation needed)
inline int getSystemThreadPriority(ThreadPriority priority) {
    switch (priority) {
        case ThreadPriority::LOW: return -5;
        case ThreadPriority::NORMAL: return 0;
        case ThreadPriority::HIGH: return 5;
        case ThreadPriority::REALTIME: return 10;
        default: return 0;
    }
}

#endif // MONITORING_TYPES_H