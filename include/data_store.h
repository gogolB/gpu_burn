#ifndef DATA_STORE_H
#define DATA_STORE_H

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>
#include <atomic>
#include "monitoring_types.h"

// Forward declarations
template<typename T> class CircularBuffer;

// Metric data wrapper for insertion
struct MetricData {
    std::string name;
    double value;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, std::string> tags;  // Optional metadata
    
    MetricData() : value(0.0), timestamp(std::chrono::system_clock::now()) {}
    MetricData(const std::string& n, double v)
        : name(n), value(v), timestamp(std::chrono::system_clock::now()) {}
    MetricData(const std::string& n, double v, const std::chrono::system_clock::time_point& t)
        : name(n), value(v), timestamp(t) {}
};

// Time-series data storage interface
class DataStore {
public:
    virtual ~DataStore() = default;
    
    // Initialize the data store
    virtual bool initialize(const RetentionPolicy& policy) = 0;
    
    // Thread-safe insertion
    virtual void insert(MetricType type, const MetricData& data) = 0;
    virtual void insert(const std::string& metricName, double value) = 0;
    virtual void insert(const std::string& metricName, const TimeSeriesDataPoint<double>& dataPoint) = 0;
    
    // Batch insertion for efficiency
    virtual void insertBatch(MetricType type, const std::vector<MetricData>& data) = 0;
    
    // Querying
    virtual TimeSeriesData query(MetricType type, const TimeRange& range) = 0;
    virtual TimeSeriesData query(const std::string& metricName, const TimeRange& range) = 0;
    virtual std::vector<TimeSeriesData> queryAll(const TimeRange& range) = 0;
    
    // Aggregation
    virtual double aggregate(MetricType type, const TimeRange& range, AggregationType aggType) = 0;
    virtual double aggregate(const std::string& metricName, const TimeRange& range, AggregationType aggType) = 0;
    
    // Latest value access
    virtual double getLatestValue(const std::string& metricName) = 0;
    virtual TimeSeriesDataPoint<double> getLatestDataPoint(const std::string& metricName) = 0;
    
    // Memory management
    virtual void setRetentionPolicy(const RetentionPolicy& policy) = 0;
    virtual void compact() = 0;
    virtual void clear() = 0;
    virtual void clearMetric(const std::string& metricName) = 0;
    
    // Statistics
    virtual size_t getMetricCount() const = 0;
    virtual size_t getTotalSamples() const = 0;
    virtual size_t getSampleCount(const std::string& metricName) const = 0;
    virtual size_t getMemoryUsage() const = 0;
    
    // Metric listing
    virtual std::vector<std::string> getMetricNames() const = 0;
    virtual std::vector<std::string> getMetricNames(MetricType type) const = 0;
    
    // Export/Import for persistence
    virtual bool exportData(const std::string& filename) = 0;
    virtual bool importData(const std::string& filename) = 0;
};

// Concrete implementation of DataStore using circular buffers
class CircularBufferDataStore : public DataStore {
public:
    CircularBufferDataStore();
    ~CircularBufferDataStore() override;
    
    // DataStore interface implementation
    bool initialize(const RetentionPolicy& policy) override;
    
    void insert(MetricType type, const MetricData& data) override;
    void insert(const std::string& metricName, double value) override;
    void insert(const std::string& metricName, const TimeSeriesDataPoint<double>& dataPoint) override;
    
    void insertBatch(MetricType type, const std::vector<MetricData>& data) override;
    
    TimeSeriesData query(MetricType type, const TimeRange& range) override;
    TimeSeriesData query(const std::string& metricName, const TimeRange& range) override;
    std::vector<TimeSeriesData> queryAll(const TimeRange& range) override;
    
    double aggregate(MetricType type, const TimeRange& range, AggregationType aggType) override;
    double aggregate(const std::string& metricName, const TimeRange& range, AggregationType aggType) override;
    
    double getLatestValue(const std::string& metricName) override;
    TimeSeriesDataPoint<double> getLatestDataPoint(const std::string& metricName) override;
    
    void setRetentionPolicy(const RetentionPolicy& policy) override;
    void compact() override;
    void clear() override;
    void clearMetric(const std::string& metricName) override;
    
    size_t getMetricCount() const override;
    size_t getTotalSamples() const override;
    size_t getSampleCount(const std::string& metricName) const override;
    size_t getMemoryUsage() const override;
    
    std::vector<std::string> getMetricNames() const override;
    std::vector<std::string> getMetricNames(MetricType type) const override;
    
    bool exportData(const std::string& filename) override;
    bool importData(const std::string& filename) override;
    
private:
    // Internal structures
    struct MetricInfo {
        MetricType type;
        std::unique_ptr<CircularBuffer<TimeSeriesDataPoint<double>>> buffer;
        std::chrono::system_clock::time_point lastUpdate;
        size_t totalInsertions;
    };
    
    // Helper methods
    void compactBuffer(MetricInfo& metric);
    TimeSeriesData queryBuffer(const MetricInfo& metric, const TimeRange& range) const;
    double aggregateBuffer(const MetricInfo& metric, const TimeRange& range, 
                          AggregationType aggType) const;
    
private:
    // Data storage
    std::unordered_map<std::string, MetricInfo> metrics_;
    mutable std::shared_mutex dataMutex_;
    
    // Configuration
    RetentionPolicy policy_;
    
    // Statistics
    std::atomic<size_t> totalSamples_{0};
    std::atomic<size_t> totalMemoryUsage_{0};
    
    // Compaction
    std::chrono::system_clock::time_point lastCompaction_;
};

// Factory function for creating data stores
std::unique_ptr<DataStore> createDataStore(const std::string& type = "circular_buffer");

#endif // DATA_STORE_H