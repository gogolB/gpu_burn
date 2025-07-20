#include "data_store.h"
#include <fstream>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <shared_mutex>

// Thread-safe circular buffer implementation
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity) 
        : capacity_(capacity), size_(0), head_(0), tail_(0) {
        buffer_.resize(capacity);
    }
    
    void push(const T& value) {
        size_t currentTail = tail_.load(std::memory_order_relaxed);
        size_t nextTail = (currentTail + 1) % capacity_;
        
        buffer_[currentTail] = value;
        tail_.store(nextTail, std::memory_order_release);
        
        size_t currentSize = size_.load(std::memory_order_relaxed);
        if (currentSize < capacity_) {
            size_.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Buffer is full, move head forward
            size_t currentHead = head_.load(std::memory_order_relaxed);
            head_.store((currentHead + 1) % capacity_, std::memory_order_release);
        }
    }
    
    std::vector<T> getRange(size_t start, size_t count) const {
        std::vector<T> result;
        result.reserve(count);
        
        size_t currentSize = size_.load(std::memory_order_acquire);
        size_t currentHead = head_.load(std::memory_order_acquire);
        
        for (size_t i = 0; i < count && i < currentSize; ++i) {
            size_t index = (currentHead + start + i) % capacity_;
            result.push_back(buffer_[index]);
        }
        
        return result;
    }
    
    std::vector<T> getAll() const {
        return getRange(0, size_.load(std::memory_order_acquire));
    }
    
    T getLatest() const {
        size_t currentSize = size_.load(std::memory_order_acquire);
        if (currentSize == 0) {
            return T{};
        }
        
        size_t currentTail = tail_.load(std::memory_order_acquire);
        size_t lastIndex = (currentTail + capacity_ - 1) % capacity_;
        return buffer_[lastIndex];
    }
    
    size_t size() const { return size_.load(std::memory_order_acquire); }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size() == 0; }
    void clear() { 
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
        size_.store(0, std::memory_order_release);
    }
    
    // Get memory usage in bytes
    size_t getMemoryUsage() const {
        return capacity_ * sizeof(T) + sizeof(CircularBuffer);
    }
    
private:
    std::vector<T> buffer_;
    size_t capacity_;
    std::atomic<size_t> size_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

// CircularBufferDataStore implementation
CircularBufferDataStore::CircularBufferDataStore() 
    : lastCompaction_(std::chrono::system_clock::now()) {
}

CircularBufferDataStore::~CircularBufferDataStore() = default;

bool CircularBufferDataStore::initialize(const RetentionPolicy& policy) {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    policy_ = policy;
    metrics_.clear();
    totalSamples_ = 0;
    totalMemoryUsage_ = 0;
    
    return true;
}

void CircularBufferDataStore::insert(MetricType type, const MetricData& data) {
    insert(data.name, TimeSeriesDataPoint<double>{data.value});
}

void CircularBufferDataStore::insert(const std::string& metricName, double value) {
    insert(metricName, TimeSeriesDataPoint<double>{value});
}

void CircularBufferDataStore::insert(const std::string& metricName, 
                                    const TimeSeriesDataPoint<double>& dataPoint) {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it == metrics_.end()) {
        // Create new metric buffer
        MetricInfo info;
        info.type = MetricType::SYSTEM;  // Default type
        info.buffer = std::make_unique<CircularBuffer<TimeSeriesDataPoint<double>>>(policy_.maxSamples);
        info.lastUpdate = std::chrono::system_clock::now();
        info.totalInsertions = 0;
        
        auto [inserted_it, success] = metrics_.emplace(metricName, std::move(info));
        it = inserted_it;
    }
    
    it->second.buffer->push(dataPoint);
    it->second.lastUpdate = std::chrono::system_clock::now();
    it->second.totalInsertions++;
    
    totalSamples_.fetch_add(1, std::memory_order_relaxed);
    
    // Check if compaction is needed
    auto now = std::chrono::system_clock::now();
    if (now - lastCompaction_ > std::chrono::minutes(5)) {
        compact();
    }
}

void CircularBufferDataStore::insertBatch(MetricType type, const std::vector<MetricData>& data) {
    for (const auto& item : data) {
        insert(type, item);
    }
}

TimeSeriesData CircularBufferDataStore::query(MetricType type, const TimeRange& range) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    // Find the first metric of the given type
    for (const auto& [name, info] : metrics_) {
        if (info.type == type) {
            return queryBuffer(info, range);
        }
    }
    
    return TimeSeriesData{};
}

TimeSeriesData CircularBufferDataStore::query(const std::string& metricName, const TimeRange& range) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end()) {
        return queryBuffer(it->second, range);
    }
    
    return TimeSeriesData{};
}

std::vector<TimeSeriesData> CircularBufferDataStore::queryAll(const TimeRange& range) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    std::vector<TimeSeriesData> results;
    results.reserve(metrics_.size());
    
    for (const auto& [name, info] : metrics_) {
        auto data = queryBuffer(info, range);
        data.metricName = name;
        results.push_back(std::move(data));
    }
    
    return results;
}

double CircularBufferDataStore::aggregate(MetricType type, const TimeRange& range, 
                                         AggregationType aggType) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    // Find the first metric of the given type
    for (const auto& [name, info] : metrics_) {
        if (info.type == type) {
            return aggregateBuffer(info, range, aggType);
        }
    }
    
    return 0.0;
}

double CircularBufferDataStore::aggregate(const std::string& metricName, const TimeRange& range,
                                         AggregationType aggType) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end()) {
        return aggregateBuffer(it->second, range, aggType);
    }
    
    return 0.0;
}

double CircularBufferDataStore::getLatestValue(const std::string& metricName) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end() && !it->second.buffer->empty()) {
        return it->second.buffer->getLatest().value;
    }
    
    return 0.0;
}

TimeSeriesDataPoint<double> CircularBufferDataStore::getLatestDataPoint(const std::string& metricName) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end() && !it->second.buffer->empty()) {
        return it->second.buffer->getLatest();
    }
    
    return TimeSeriesDataPoint<double>{};
}

void CircularBufferDataStore::setRetentionPolicy(const RetentionPolicy& policy) {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    policy_ = policy;
    
    // Resize existing buffers if needed
    for (auto& [name, info] : metrics_) {
        if (info.buffer->capacity() != policy.maxSamples) {
            // Create new buffer with new capacity
            auto newBuffer = std::make_unique<CircularBuffer<TimeSeriesDataPoint<double>>>(policy.maxSamples);
            
            // Copy existing data
            auto allData = info.buffer->getAll();
            for (const auto& point : allData) {
                newBuffer->push(point);
            }
            
            info.buffer = std::move(newBuffer);
        }
    }
}

void CircularBufferDataStore::compact() {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    auto now = std::chrono::system_clock::now();
    
    // Remove old data based on retention policy
    for (auto& [name, info] : metrics_) {
        compactBuffer(info);
    }
    
    lastCompaction_ = now;
}

void CircularBufferDataStore::clear() {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    metrics_.clear();
    totalSamples_ = 0;
    totalMemoryUsage_ = 0;
}

void CircularBufferDataStore::clearMetric(const std::string& metricName) {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end()) {
        size_t samples = it->second.buffer->size();
        totalSamples_ -= samples;
        metrics_.erase(it);
    }
}

size_t CircularBufferDataStore::getMetricCount() const {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    return metrics_.size();
}

size_t CircularBufferDataStore::getTotalSamples() const {
    return totalSamples_.load(std::memory_order_acquire);
}

size_t CircularBufferDataStore::getSampleCount(const std::string& metricName) const {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    auto it = metrics_.find(metricName);
    if (it != metrics_.end()) {
        return it->second.buffer->size();
    }
    
    return 0;
}

size_t CircularBufferDataStore::getMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    size_t totalUsage = sizeof(CircularBufferDataStore);
    for (const auto& [name, info] : metrics_) {
        totalUsage += name.length() + sizeof(MetricInfo) + info.buffer->getMemoryUsage();
    }
    
    return totalUsage;
}

std::vector<std::string> CircularBufferDataStore::getMetricNames() const {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    std::vector<std::string> names;
    names.reserve(metrics_.size());
    
    for (const auto& [name, info] : metrics_) {
        names.push_back(name);
    }
    
    return names;
}

std::vector<std::string> CircularBufferDataStore::getMetricNames(MetricType type) const {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    std::vector<std::string> names;
    
    for (const auto& [name, info] : metrics_) {
        if (info.type == type) {
            names.push_back(name);
        }
    }
    
    return names;
}

bool CircularBufferDataStore::exportData(const std::string& filename) {
    std::shared_lock<std::shared_mutex> lock(dataMutex_);
    
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        
        // Write header
        size_t metricCount = metrics_.size();
        file.write(reinterpret_cast<const char*>(&metricCount), sizeof(metricCount));
        
        // Write each metric
        for (const auto& [name, info] : metrics_) {
            // Write metric name
            size_t nameLen = name.length();
            file.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
            file.write(name.c_str(), nameLen);
            
            // Write metric type
            file.write(reinterpret_cast<const char*>(&info.type), sizeof(info.type));
            
            // Write data points
            auto allData = info.buffer->getAll();
            size_t dataCount = allData.size();
            file.write(reinterpret_cast<const char*>(&dataCount), sizeof(dataCount));
            
            for (const auto& point : allData) {
                auto timeT = std::chrono::system_clock::to_time_t(point.timestamp);
                file.write(reinterpret_cast<const char*>(&timeT), sizeof(timeT));
                file.write(reinterpret_cast<const char*>(&point.value), sizeof(point.value));
            }
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool CircularBufferDataStore::importData(const std::string& filename) {
    std::unique_lock<std::shared_mutex> lock(dataMutex_);
    
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        
        clear();
        
        // Read header
        size_t metricCount;
        file.read(reinterpret_cast<char*>(&metricCount), sizeof(metricCount));
        
        // Read each metric
        for (size_t i = 0; i < metricCount; ++i) {
            // Read metric name
            size_t nameLen;
            file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
            
            std::string name(nameLen, '\0');
            file.read(&name[0], nameLen);
            
            // Read metric type
            MetricType type;
            file.read(reinterpret_cast<char*>(&type), sizeof(type));
            
            // Create metric
            MetricInfo info;
            info.type = type;
            info.buffer = std::make_unique<CircularBuffer<TimeSeriesDataPoint<double>>>(policy_.maxSamples);
            info.lastUpdate = std::chrono::system_clock::now();
            info.totalInsertions = 0;
            
            // Read data points
            size_t dataCount;
            file.read(reinterpret_cast<char*>(&dataCount), sizeof(dataCount));
            
            for (size_t j = 0; j < dataCount; ++j) {
                std::time_t timeT;
                double value;
                file.read(reinterpret_cast<char*>(&timeT), sizeof(timeT));
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                
                TimeSeriesDataPoint<double> point;
                point.timestamp = std::chrono::system_clock::from_time_t(timeT);
                point.value = value;
                
                info.buffer->push(point);
                info.totalInsertions++;
            }
            
            metrics_[name] = std::move(info);
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void CircularBufferDataStore::compactBuffer(MetricInfo& metric) {
    if (!policy_.compressOldData) return;
    
    auto now = std::chrono::system_clock::now();
    auto allData = metric.buffer->getAll();
    
    if (allData.size() < policy_.maxSamples * 0.9) {
        return;  // No need to compact yet
    }
    
    // Keep only data within retention period
    std::vector<TimeSeriesDataPoint<double>> filteredData;
    filteredData.reserve(allData.size());
    
    for (const auto& point : allData) {
        auto age = now - point.timestamp;
        if (age <= policy_.maxAge) {
            filteredData.push_back(point);
        }
    }
    
    // If still too many samples, apply compression
    if (filteredData.size() > policy_.maxSamples * policy_.compressionRatio) {
        size_t targetSize = static_cast<size_t>(policy_.maxSamples * policy_.compressionRatio);
        size_t step = filteredData.size() / targetSize;
        
        std::vector<TimeSeriesDataPoint<double>> compressedData;
        compressedData.reserve(targetSize);
        
        for (size_t i = 0; i < filteredData.size(); i += step) {
            compressedData.push_back(filteredData[i]);
        }
        
        filteredData = std::move(compressedData);
    }
    
    // Rebuild buffer with filtered data
    metric.buffer->clear();
    for (const auto& point : filteredData) {
        metric.buffer->push(point);
    }
}

TimeSeriesData CircularBufferDataStore::queryBuffer(const MetricInfo& metric, 
                                                   const TimeRange& range) const {
    TimeSeriesData result;
    result.type = metric.type;
    
    auto allData = metric.buffer->getAll();
    
    for (const auto& point : allData) {
        if (point.timestamp >= range.start && point.timestamp <= range.end) {
            result.dataPoints.push_back(point);
        }
    }
    
    result.calculateStats();
    return result;
}

double CircularBufferDataStore::aggregateBuffer(const MetricInfo& metric, const TimeRange& range,
                                               AggregationType aggType) const {
    auto data = queryBuffer(metric, range);
    
    if (data.dataPoints.empty()) {
        return 0.0;
    }
    
    switch (aggType) {
        case AggregationType::MIN:
            return data.stats.min;
            
        case AggregationType::MAX:
            return data.stats.max;
            
        case AggregationType::AVERAGE:
            return data.stats.average;
            
        case AggregationType::SUM: {
            double sum = 0.0;
            for (const auto& point : data.dataPoints) {
                sum += point.value;
            }
            return sum;
        }
        
        case AggregationType::COUNT:
            return static_cast<double>(data.dataPoints.size());
            
        case AggregationType::PERCENTILE_50:
        case AggregationType::PERCENTILE_95:
        case AggregationType::PERCENTILE_99: {
            std::vector<double> values;
            values.reserve(data.dataPoints.size());
            for (const auto& point : data.dataPoints) {
                values.push_back(point.value);
            }
            std::sort(values.begin(), values.end());
            
            double percentile = 0.5;
            if (aggType == AggregationType::PERCENTILE_95) percentile = 0.95;
            else if (aggType == AggregationType::PERCENTILE_99) percentile = 0.99;
            
            size_t index = static_cast<size_t>(values.size() * percentile);
            return values[std::min(index, values.size() - 1)];
        }
        
        default:
            return 0.0;
    }
}

// Factory function implementation
std::unique_ptr<DataStore> createDataStore(const std::string& type) {
    if (type == "circular_buffer" || type.empty()) {
        return std::make_unique<CircularBufferDataStore>();
    }
    
    // Add other data store types here in the future
    
    return nullptr;
}