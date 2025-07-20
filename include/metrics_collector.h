#ifndef METRICS_COLLECTOR_H
#define METRICS_COLLECTOR_H

#include <string>
#include <memory>
#include <chrono>
#include "monitoring_types.h"

// Forward declaration
class MonitoringEngine;

// Abstract base class for all metrics collectors
class MetricsCollector {
public:
    virtual ~MetricsCollector() = default;
    
    // Lifecycle management
    virtual bool initialize(const CollectorConfig& config) = 0;
    virtual void cleanup() = 0;
    
    // Collection
    virtual MetricResult collect() = 0;
    virtual bool isSupported(int deviceId) const = 0;
    
    // Metadata
    virtual std::string getName() const = 0;
    virtual MetricType getType() const = 0;
    virtual std::chrono::milliseconds getCollectionInterval() const = 0;
    
    // Enable/disable collection
    virtual void enable(bool enabled) { enabled_ = enabled; }
    virtual bool isEnabled() const { return enabled_; }
    
    // Error handling
    virtual std::string getLastError() const { return lastError_; }
    virtual size_t getErrorCount() const { return errorCount_; }
    virtual void resetErrorCount() { errorCount_ = 0; }
    
    // Parent engine reference
    void setMonitoringEngine(MonitoringEngine* engine) { engine_ = engine; }
    MonitoringEngine* getMonitoringEngine() const { return engine_; }
    
    // Device management
    virtual void setDeviceId(int deviceId) { deviceId_ = deviceId; }
    virtual int getDeviceId() const { return deviceId_; }
    
protected:
    // Helper methods for derived classes
    void setLastError(const std::string& error) { 
        lastError_ = error; 
        errorCount_++;
    }
    
    void clearLastError() { 
        lastError_.clear(); 
    }
    
    // Create a failed result with error message
    MetricResult createFailedResult(const std::string& error) {
        MetricResult result;
        result.success = false;
        result.type = getType();
        result.errorMessage = error;
        result.timestamp = std::chrono::system_clock::now();
        return result;
    }
    
    // Create a successful result
    MetricResult createSuccessResult() {
        MetricResult result;
        result.success = true;
        result.type = getType();
        result.timestamp = std::chrono::system_clock::now();
        return result;
    }
    
protected:
    MonitoringEngine* engine_ = nullptr;
    bool enabled_ = true;
    std::string lastError_;
    size_t errorCount_ = 0;
    int deviceId_ = 0;
    CollectorConfig config_;
};

// Factory function type for creating collectors
using CollectorFactory = std::unique_ptr<MetricsCollector>(*)();

// Helper macro for collector registration
#define REGISTER_COLLECTOR(CollectorClass) \
    std::unique_ptr<MetricsCollector> create##CollectorClass() { \
        return std::make_unique<CollectorClass>(); \
    }

#endif // METRICS_COLLECTOR_H