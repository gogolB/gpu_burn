#include "report_generator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>

// CSV Reporter - tabular format for analysis
class CSVReporter : public ReportGenerator {
public:
    CSVReporter() {
        includeHeaders_ = true;
        includeTimestamp_ = true;
        delimiter_ = ",";
        floatPrecision_ = 2;
    }
    
    void generate(const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        generate(std::cout, snapshot, historical);
    }
    
    void generate(std::ostream& output,
                 const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        // Generate snapshot CSV
        if (includeHeaders_ && !headerWritten_) {
            writeHeaders(output);
            headerWritten_ = true;
        }
        writeSnapshotRow(output, snapshot);
        
        // If historical data is available, write it separately
        if (!historical.dataPoints.empty()) {
            output << "\n\n";  // Separator
            writeHistoricalData(output, historical);
        }
        
        output.flush();
    }
    
    bool generateToFile(const std::string& filename,
                       const MonitoringSnapshot& snapshot,
                       const TimeSeriesData& historical) override {
        std::ofstream file(filename, std::ios::app);  // Append mode for continuous logging
        if (!file.is_open()) return false;
        
        // Check if file is empty to write headers
        file.seekp(0, std::ios::end);
        bool isEmpty = (file.tellp() == 0);
        file.seekp(0, std::ios::beg);
        
        if (isEmpty && includeHeaders_) {
            writeHeaders(file);
        }
        
        writeSnapshotRow(file, snapshot);
        
        // Write historical data to a separate file if requested
        if (!historical.dataPoints.empty()) {
            std::string histFile = filename.substr(0, filename.rfind('.')) + "_historical.csv";
            std::ofstream histStream(histFile);
            if (histStream.is_open()) {
                writeHistoricalData(histStream, historical);
            }
        }
        
        return true;
    }
    
    std::string getFormat() const override { return "csv"; }
    
    void configure(const std::unordered_map<std::string, std::string>& options) override {
        auto it = options.find("headers");
        if (it != options.end()) {
            includeHeaders_ = (it->second == "true" || it->second == "1");
        }
        
        it = options.find("delimiter");
        if (it != options.end()) {
            delimiter_ = it->second;
        }
        
        it = options.find("precision");
        if (it != options.end()) {
            floatPrecision_ = std::stoi(it->second);
        }
    }
    
private:
    void writeHeaders(std::ostream& out) {
        if (includeTimestamp_) {
            out << "timestamp" << delimiter_ << "timestamp_epoch" << delimiter_;
        }
        
        // Health headers
        out << "temperature_celsius" << delimiter_
            << "power_watts" << delimiter_
            << "core_clock_mhz" << delimiter_
            << "memory_clock_mhz" << delimiter_
            << "fan_speed_percent" << delimiter_;
        
        // Performance headers
        out << "current_gflops" << delimiter_
            << "average_gflops" << delimiter_
            << "peak_gflops" << delimiter_
            << "bandwidth_gbps" << delimiter_
            << "sm_efficiency_percent" << delimiter_;
        
        // Error headers
        out << "total_errors" << delimiter_
            << "recent_errors" << delimiter_
            << "error_rate" << delimiter_;
        
        // Status headers
        out << "is_throttling" << delimiter_
            << "throttle_reasons" << delimiter_
            << "active_alerts";
        
        out << "\n";
    }
    
    void writeSnapshotRow(std::ostream& out, const MonitoringSnapshot& snapshot) {
        if (includeTimestamp_) {
            out << formatTimestamp(snapshot.timestamp) << delimiter_
                << toEpochMillis(snapshot.timestamp) << delimiter_;
        }
        
        // Set float precision
        out << std::fixed << std::setprecision(floatPrecision_);
        
        // Health data
        out << snapshot.health.temperature << delimiter_
            << snapshot.health.power << delimiter_
            << snapshot.health.coreClockMHz << delimiter_
            << snapshot.health.memoryClockMHz << delimiter_
            << snapshot.health.fanSpeedPercent << delimiter_;
        
        // Performance data
        out << snapshot.performance.currentGFLOPS << delimiter_
            << snapshot.performance.avgGFLOPS << delimiter_
            << snapshot.performance.peakGFLOPS << delimiter_
            << snapshot.performance.bandwidthGBps << delimiter_
            << snapshot.performance.smEfficiency << delimiter_;
        
        // Error data
        out << snapshot.errors.totalErrors << delimiter_
            << snapshot.errors.recentErrors << delimiter_;
        
        // Calculate error rate
        double errorRate = 0.0;
        if (snapshot.errors.totalErrors > 0) {
            // This is a simplified calculation
            errorRate = static_cast<double>(snapshot.errors.recentErrors) / 60.0;  // Per second
        }
        out << errorRate << delimiter_;
        
        // Status data
        out << (snapshot.status.isThrottling ? "1" : "0") << delimiter_;
        
        // Throttle reasons (semicolon-separated list)
        if (!snapshot.status.throttleReasons.empty()) {
            out << "\"";
            for (size_t i = 0; i < snapshot.status.throttleReasons.size(); ++i) {
                out << escapeCSV(snapshot.status.throttleReasons[i]);
                if (i < snapshot.status.throttleReasons.size() - 1) out << ";";
            }
            out << "\"";
        }
        out << delimiter_;
        
        // Active alerts (semicolon-separated list)
        if (!snapshot.status.activeAlerts.empty()) {
            out << "\"";
            for (size_t i = 0; i < snapshot.status.activeAlerts.size(); ++i) {
                out << escapeCSV(snapshot.status.activeAlerts[i]);
                if (i < snapshot.status.activeAlerts.size() - 1) out << ";";
            }
            out << "\"";
        }
        
        out << "\n";
    }
    
    void writeHistoricalData(std::ostream& out, const TimeSeriesData& historical) {
        // Header for historical data
        out << "# Historical Data: " << historical.metricName << "\n";
        out << "timestamp" << delimiter_ << "timestamp_epoch" << delimiter_ << "value\n";
        
        // Write data points
        for (const auto& point : historical.dataPoints) {
            out << formatTimestamp(point.timestamp) << delimiter_
                << toEpochMillis(point.timestamp) << delimiter_
                << std::fixed << std::setprecision(floatPrecision_) << point.value << "\n";
        }
        
        // Write statistics
        const_cast<TimeSeriesData&>(historical).calculateStats();
        out << "\n# Statistics\n";
        out << "metric" << delimiter_ << "value\n";
        out << "min" << delimiter_ << historical.stats.min << "\n";
        out << "max" << delimiter_ << historical.stats.max << "\n";
        out << "average" << delimiter_ << historical.stats.average << "\n";
        out << "stddev" << delimiter_ << historical.stats.stddev << "\n";
        out << "count" << delimiter_ << historical.stats.count << "\n";
    }
    
    // Helper functions
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    long long toEpochMillis(const std::chrono::system_clock::time_point& tp) const {
        auto duration = tp.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }
    
    std::string escapeCSV(const std::string& str) const {
        // Escape quotes and handle special characters
        std::string escaped;
        for (char c : str) {
            if (c == '"') {
                escaped += "\"\"";  // Double quotes
            } else {
                escaped += c;
            }
        }
        return escaped;
    }
    
private:
    bool includeHeaders_;
    bool includeTimestamp_;
    bool headerWritten_ = false;
    std::string delimiter_;
    int floatPrecision_;
};

// Factory function
REGISTER_REPORT_GENERATOR(CSVReporter)