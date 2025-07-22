#include "metrics_collector.h"
#include "monitoring_types.h"
#include "validation_types.h"
#include <mutex>
#include <deque>
#include <unordered_map>
#include <cmath>

// Error pattern analyzer for detecting error clusters and patterns
class ErrorPatternAnalyzer {
public:
    struct ErrorEvent {
        std::chrono::system_clock::time_point timestamp;
        std::string errorType;
        size_t location;  // Memory location or kernel ID
        double temperature;  // Temperature at time of error
        double power;       // Power at time of error
    };
    
    void addError(const ErrorEvent& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        errors_.push_back(error);
        
        // Keep only recent errors (last hour)
        auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(1);
        while (!errors_.empty() && errors_.front().timestamp < cutoff) {
            errors_.pop_front();
        }
        
        // Update pattern analysis
        updatePatterns();
    }
    
    bool hasCluster(std::chrono::seconds window = std::chrono::seconds(60)) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (errors_.size() < 3) return false;
        
        auto now = std::chrono::system_clock::now();
        int count = 0;
        
        for (const auto& error : errors_) {
            if (now - error.timestamp <= window) {
                count++;
            }
        }
        
        return count >= 3;  // 3 or more errors in window indicates cluster
    }
    
    double getErrorRate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (errors_.empty()) return 0.0;
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            errors_.back().timestamp - errors_.front().timestamp).count();
        
        if (duration == 0) return 0.0;
        
        return static_cast<double>(errors_.size()) / duration;
    }
    
    std::unordered_map<std::string, size_t> getErrorDistribution() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::unordered_map<std::string, size_t> distribution;
        for (const auto& error : errors_) {
            distribution[error.errorType]++;
        }
        
        return distribution;
    }
    
    double getTemperatureCorrelation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (errors_.size() < 2) return 0.0;
        
        // Calculate correlation between temperature and error frequency
        std::vector<std::pair<double, int>> tempErrorCounts;
        
        // Group errors by temperature ranges
        for (const auto& error : errors_) {
            int tempBucket = static_cast<int>(error.temperature / 5.0) * 5;  // 5-degree buckets
            bool found = false;
            
            for (auto& pair : tempErrorCounts) {
                if (std::abs(pair.first - tempBucket) < 2.5) {
                    pair.second++;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                tempErrorCounts.push_back({tempBucket, 1});
            }
        }
        
        // Calculate correlation coefficient
        if (tempErrorCounts.size() < 2) return 0.0;
        
        double meanTemp = 0.0, meanErrors = 0.0;
        for (const auto& pair : tempErrorCounts) {
            meanTemp += pair.first;
            meanErrors += pair.second;
        }
        meanTemp /= tempErrorCounts.size();
        meanErrors /= tempErrorCounts.size();
        
        double numerator = 0.0, denomTemp = 0.0, denomErrors = 0.0;
        for (const auto& pair : tempErrorCounts) {
            double tempDiff = pair.first - meanTemp;
            double errorDiff = pair.second - meanErrors;
            numerator += tempDiff * errorDiff;
            denomTemp += tempDiff * tempDiff;
            denomErrors += errorDiff * errorDiff;
        }
        
        if (denomTemp == 0.0 || denomErrors == 0.0) return 0.0;
        
        return numerator / (std::sqrt(denomTemp) * std::sqrt(denomErrors));
    }
    
private:
    void updatePatterns() {
        // Analyze for common patterns
        // This is a placeholder - real implementation would use more sophisticated pattern detection
    }
    
    mutable std::mutex mutex_;
    std::deque<ErrorEvent> errors_;
};

// Error Monitor - tracks SDC detection patterns and error statistics
class ErrorMonitor : public MetricsCollector {
public:
    ErrorMonitor() : totalErrors_(0), recentWindowMinutes_(1) {
        name_ = "ErrorMonitor";
        type_ = MetricType::ERROR;
        collectionInterval_ = std::chrono::milliseconds(100);
    }
    
    ~ErrorMonitor() override {
        cleanup();
    }
    
    bool initialize(const CollectorConfig& config) override {
        config_ = config;
        
        // Initialize error categories
        errorStats_["SDC_DETECTED"] = ErrorStats{};
        errorStats_["SDC_CORRECTED"] = ErrorStats{};
        errorStats_["VALIDATION_FAILED"] = ErrorStats{};
        errorStats_["MEMORY_ERROR"] = ErrorStats{};
        errorStats_["TIMEOUT"] = ErrorStats{};
        errorStats_["KERNEL_FAILURE"] = ErrorStats{};
        
        analyzer_ = std::make_unique<ErrorPatternAnalyzer>();
        
        return true;
    }
    
    void cleanup() override {
        // Nothing specific to clean up
    }
    
    MetricResult collect() override {
        MetricResult result = createSuccessResult();
        
        std::lock_guard<std::mutex> lock(statsMutex_);
        
        // Total errors
        result.values["error.total"] = static_cast<double>(totalErrors_);
        
        // Recent errors (last minute)
        size_t recentErrors = 0;
        auto cutoff = std::chrono::steady_clock::now() - std::chrono::minutes(recentWindowMinutes_);
        
        for (const auto& [type, stats] : errorStats_) {
            for (const auto& timestamp : stats.timestamps) {
                if (timestamp > cutoff) {
                    recentErrors++;
                }
            }
            
            // Per-type statistics
            result.values["error." + type + ".count"] = static_cast<double>(stats.count);
            result.values["error." + type + ".rate"] = stats.getRate();
        }
        
        result.values["error.recent"] = static_cast<double>(recentErrors);
        
        // Pattern analysis
        result.values["error.has_cluster"] = analyzer_->hasCluster() ? 1.0 : 0.0;
        result.values["error.rate"] = analyzer_->getErrorRate();
        result.values["error.temp_correlation"] = analyzer_->getTemperatureCorrelation();
        
        // Error distribution
        auto distribution = analyzer_->getErrorDistribution();
        for (const auto& [type, count] : distribution) {
            result.values["error.distribution." + type] = static_cast<double>(count);
        }
        
        return result;
    }
    bool isSupported(int /* deviceId */) const override {
        // Error monitoring is always supported
        return true;
    }
    
    std::string getName() const override { return name_; }
    MetricType getType() const override { return type_; }
    std::chrono::milliseconds getCollectionInterval() const override { return collectionInterval_; }
    
    // Record an error event
    void recordError(const std::string& errorType, double temperature = 0.0, double power = 0.0) {
        std::lock_guard<std::mutex> lock(statsMutex_);
        
        auto now = std::chrono::steady_clock::now();
        auto systemNow = std::chrono::system_clock::now();
        
        // Update statistics
        auto it = errorStats_.find(errorType);
        if (it != errorStats_.end()) {
            it->second.count++;
            it->second.timestamps.push_back(now);
            it->second.lastOccurrence = now;
            
            // Keep only recent timestamps
            auto cutoff = now - std::chrono::hours(1);
            it->second.timestamps.erase(
                std::remove_if(it->second.timestamps.begin(), it->second.timestamps.end(),
                    [cutoff](const auto& ts) { return ts < cutoff; }),
                it->second.timestamps.end()
            );
        } else {
            ErrorStats stats;
            stats.count = 1;
            stats.timestamps.push_back(now);
            stats.lastOccurrence = now;
            errorStats_[errorType] = stats;
        }
        
        totalErrors_++;
        
        // Add to pattern analyzer
        ErrorPatternAnalyzer::ErrorEvent event;
        event.timestamp = systemNow;
        event.errorType = errorType;
        event.location = 0;  // Would be set based on actual error location
        event.temperature = temperature;
        event.power = power;
        analyzer_->addError(event);
    }
    
    // Record validation result
    void recordValidationResult(const ValidationResult& result) {
        if (!result.passed) {
            std::string errorType = "VALIDATION_FAILED_" + getValidationTypeName(result.method);
            recordError(errorType);
            
            if (result.corruptedElements > 0) {
                recordError("SDC_DETECTED");
            }
        }
    }
    
    // Get error statistics
    std::unordered_map<std::string, size_t> getErrorCounts() const {
        std::lock_guard<std::mutex> lock(statsMutex_);
        
        std::unordered_map<std::string, size_t> counts;
        for (const auto& [type, stats] : errorStats_) {
            counts[type] = stats.count;
        }
        
        return counts;
    }
    
private:
    struct ErrorStats {
        size_t count = 0;
        std::vector<std::chrono::steady_clock::time_point> timestamps;
        std::chrono::steady_clock::time_point lastOccurrence;
        
        double getRate() const {
            if (timestamps.size() < 2) return 0.0;
            
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                timestamps.back() - timestamps.front()).count();
            
            if (duration == 0) return 0.0;
            
            return static_cast<double>(timestamps.size()) / duration;
        }
    };
    
    std::unique_ptr<ErrorPatternAnalyzer> analyzer_;
    std::unordered_map<std::string, ErrorStats> errorStats_;
    mutable std::mutex statsMutex_;
    std::atomic<size_t> totalErrors_;
    int recentWindowMinutes_;
    
    std::string name_;
    MetricType type_;
    std::chrono::milliseconds collectionInterval_;
};

// Factory function
REGISTER_COLLECTOR(ErrorMonitor)