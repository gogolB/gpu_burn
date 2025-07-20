#include "report_generator.h"
#include <sstream>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <numeric>

// Implementation of ReportGenerator helper methods
std::string ReportGenerator::formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
    auto time_t = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

std::string ReportGenerator::formatDuration(const std::chrono::duration<double>& duration) const {
    auto seconds = duration.count();
    
    if (seconds < 1.0) {
        return std::to_string(static_cast<int>(seconds * 1000)) + "ms";
    } else if (seconds < 60.0) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << seconds << "s";
        return ss.str();
    } else if (seconds < 3600.0) {
        int minutes = static_cast<int>(seconds / 60);
        int remainingSeconds = static_cast<int>(seconds) % 60;
        return std::to_string(minutes) + "m " + std::to_string(remainingSeconds) + "s";
    } else {
        int hours = static_cast<int>(seconds / 3600);
        int minutes = static_cast<int>((seconds - hours * 3600) / 60);
        return std::to_string(hours) + "h " + std::to_string(minutes) + "m";
    }
}

std::string ReportGenerator::formatBytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unitIndex = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unitIndex < 4) {
        size /= 1024.0;
        unitIndex++;
    }
    
    std::stringstream ss;
    if (unitIndex == 0) {
        ss << static_cast<int>(size) << " " << units[unitIndex];
    } else {
        ss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
    }
    
    return ss.str();
}

std::string ReportGenerator::formatPercentage(double value) const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << value << "%";
    return ss.str();
}

std::string ReportGenerator::formatMetricValue(double value, const std::string& unit) const {
    std::stringstream ss;
    
    // Choose appropriate precision based on value magnitude
    if (std::abs(value) < 0.01) {
        ss << std::scientific << std::setprecision(2) << value;
    } else if (std::abs(value) < 10.0) {
        ss << std::fixed << std::setprecision(3) << value;
    } else if (std::abs(value) < 100.0) {
        ss << std::fixed << std::setprecision(2) << value;
    } else if (std::abs(value) < 1000.0) {
        ss << std::fixed << std::setprecision(1) << value;
    } else {
        ss << std::fixed << std::setprecision(0) << value;
    }
    
    if (!unit.empty()) {
        ss << " " << unit;
    }
    
    return ss.str();
}

// Utility namespace for monitoring-specific functions
namespace MonitoringUtils {
    
    // Calculate percentile from a sorted vector
    double calculatePercentile(const std::vector<double>& sortedValues, double percentile) {
        if (sortedValues.empty()) return 0.0;
        
        double index = (percentile / 100.0) * (sortedValues.size() - 1);
        size_t lowerIndex = static_cast<size_t>(index);
        size_t upperIndex = lowerIndex + 1;
        
        if (upperIndex >= sortedValues.size()) {
            return sortedValues.back();
        }
        
        double lowerValue = sortedValues[lowerIndex];
        double upperValue = sortedValues[upperIndex];
        double fraction = index - lowerIndex;
        
        return lowerValue + fraction * (upperValue - lowerValue);
    }
    
    // Calculate moving average
    std::vector<double> calculateMovingAverage(const std::vector<double>& values, size_t windowSize) {
        if (values.empty() || windowSize == 0) return {};
        
        std::vector<double> result;
        result.reserve(values.size());
        
        for (size_t i = 0; i < values.size(); ++i) {
            size_t start = (i >= windowSize - 1) ? i - windowSize + 1 : 0;
            size_t count = i - start + 1;
            
            double sum = std::accumulate(values.begin() + start, values.begin() + i + 1, 0.0);
            result.push_back(sum / count);
        }
        
        return result;
    }
    
    // Detect anomalies using z-score
    std::vector<bool> detectAnomalies(const std::vector<double>& values, double threshold = 3.0) {
        if (values.size() < 2) return std::vector<bool>(values.size(), false);
        
        // Calculate mean and standard deviation
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        
        double variance = 0.0;
        for (double val : values) {
            variance += (val - mean) * (val - mean);
        }
        variance /= values.size();
        double stddev = std::sqrt(variance);
        
        // Mark anomalies
        std::vector<bool> anomalies;
        anomalies.reserve(values.size());
        
        for (double val : values) {
            double zScore = (stddev > 0) ? std::abs((val - mean) / stddev) : 0.0;
            anomalies.push_back(zScore > threshold);
        }
        
        return anomalies;
    }
    
    // Format large numbers with suffixes (K, M, G, T)
    std::string formatLargeNumber(double value) {
        const char* suffixes[] = {"", "K", "M", "G", "T", "P"};
        int suffixIndex = 0;
        
        while (std::abs(value) >= 1000.0 && suffixIndex < 5) {
            value /= 1000.0;
            suffixIndex++;
        }
        
        std::stringstream ss;
        if (suffixIndex == 0) {
            ss << static_cast<long long>(value);
        } else if (std::abs(value) < 10.0) {
            ss << std::fixed << std::setprecision(2) << value << suffixes[suffixIndex];
        } else if (std::abs(value) < 100.0) {
            ss << std::fixed << std::setprecision(1) << value << suffixes[suffixIndex];
        } else {
            ss << std::fixed << std::setprecision(0) << value << suffixes[suffixIndex];
        }
        
        return ss.str();
    }
    
    // Calculate rate of change
    double calculateRateOfChange(double oldValue, double newValue, double timeDeltaSeconds) {
        if (timeDeltaSeconds <= 0) return 0.0;
        return (newValue - oldValue) / timeDeltaSeconds;
    }
    
    // Smooth data using exponential moving average
    std::vector<double> exponentialMovingAverage(const std::vector<double>& values, double alpha) {
        if (values.empty()) return {};
        
        std::vector<double> result;
        result.reserve(values.size());
        
        result.push_back(values[0]);
        for (size_t i = 1; i < values.size(); ++i) {
            double ema = alpha * values[i] + (1 - alpha) * result.back();
            result.push_back(ema);
        }
        
        return result;
    }
    
    // Check if value is within threshold
    bool isWithinThreshold(double value, double threshold, double tolerance = 0.0) {
        return value <= (threshold + tolerance);
    }
    
    // Calculate trend direction
    enum class TrendDirection { INCREASING, DECREASING, STABLE };
    
    TrendDirection calculateTrend(const std::vector<double>& values, size_t windowSize = 5) {
        if (values.size() < windowSize) return TrendDirection::STABLE;
        
        // Calculate linear regression over the window
        size_t start = values.size() - windowSize;
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        
        for (size_t i = 0; i < windowSize; ++i) {
            double x = static_cast<double>(i);
            double y = values[start + i];
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumX2 += x * x;
        }
        
        double n = static_cast<double>(windowSize);
        double slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        
        // Determine trend based on slope
        if (slope > 0.01) return TrendDirection::INCREASING;
        if (slope < -0.01) return TrendDirection::DECREASING;
        return TrendDirection::STABLE;
    }
    
    // Parse duration string (e.g., "5m", "1h30m", "30s")
    std::chrono::seconds parseDuration(const std::string& durationStr) {
        std::chrono::seconds total{0};
        std::string current;
        
        for (char c : durationStr) {
            if (std::isdigit(c)) {
                current += c;
            } else if (!current.empty()) {
                int value = std::stoi(current);
                switch (c) {
                    case 's': total += std::chrono::seconds(value); break;
                    case 'm': total += std::chrono::minutes(value); break;
                    case 'h': total += std::chrono::hours(value); break;
                    case 'd': total += std::chrono::hours(value * 24); break;
                    default: break;
                }
                current.clear();
            }
        }
        
        // Handle case where duration ends with a number (assume seconds)
        if (!current.empty()) {
            total += std::chrono::seconds(std::stoi(current));
        }
        
        return total;
    }
    
    // Generate histogram bins
    struct HistogramBin {
        double min;
        double max;
        size_t count;
    };
    
    std::vector<HistogramBin> generateHistogram(const std::vector<double>& values, size_t numBins) {
        if (values.empty() || numBins == 0) return {};
        
        auto minmax = std::minmax_element(values.begin(), values.end());
        double min = *minmax.first;
        double max = *minmax.second;
        double binWidth = (max - min) / numBins;
        
        std::vector<HistogramBin> bins(numBins);
        for (size_t i = 0; i < numBins; ++i) {
            bins[i].min = min + i * binWidth;
            bins[i].max = min + (i + 1) * binWidth;
            bins[i].count = 0;
        }
        
        // Count values in each bin
        for (double value : values) {
            size_t binIndex = static_cast<size_t>((value - min) / binWidth);
            if (binIndex >= numBins) binIndex = numBins - 1;
            bins[binIndex].count++;
        }
        
        return bins;
    }
    
    // Calculate correlation coefficient
    double calculateCorrelation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.size() < 2) return 0.0;
        
        double n = static_cast<double>(x.size());
        double sumX = std::accumulate(x.begin(), x.end(), 0.0);
        double sumY = std::accumulate(y.begin(), y.end(), 0.0);
        double sumXY = 0.0, sumX2 = 0.0, sumY2 = 0.0;
        
        for (size_t i = 0; i < x.size(); ++i) {
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }
        
        double numerator = n * sumXY - sumX * sumY;
        double denominator = std::sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        if (denominator == 0) return 0.0;
        return numerator / denominator;
    }
}