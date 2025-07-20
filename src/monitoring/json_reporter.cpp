#include "report_generator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>

// JSON Reporter - structured JSON output for monitoring data
class JSONReporter : public ReportGenerator {
public:
    JSONReporter() {
        prettyPrint_ = true;
        includeTimestamps_ = true;
        includeHistorical_ = true;
    }
    
    void generate(const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        generate(std::cout, snapshot, historical);
    }
    
    void generate(std::ostream& output,
                 const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        output << generateJSON(snapshot, historical);
        output.flush();
    }
    
    bool generateToFile(const std::string& filename,
                       const MonitoringSnapshot& snapshot,
                       const TimeSeriesData& historical) override {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        file << generateJSON(snapshot, historical);
        return true;
    }
    
    std::string getFormat() const override { return "json"; }
    
    void configure(const std::unordered_map<std::string, std::string>& options) override {
        auto it = options.find("pretty");
        if (it != options.end()) {
            prettyPrint_ = (it->second == "true" || it->second == "1");
        }
        
        it = options.find("timestamps");
        if (it != options.end()) {
            includeTimestamps_ = (it->second == "true" || it->second == "1");
        }
        
        it = options.find("historical");
        if (it != options.end()) {
            includeHistorical_ = (it->second == "true" || it->second == "1");
        }
    }
    
private:
    std::string generateJSON(const MonitoringSnapshot& snapshot, 
                            const TimeSeriesData& historical) {
        std::stringstream json;
        std::string indent = prettyPrint_ ? "  " : "";
        std::string newline = prettyPrint_ ? "\n" : "";
        
        json << "{" << newline;
        
        // Metadata
        json << indent << "\"metadata\": {" << newline;
        json << indent << indent << "\"format_version\": \"1.0\"," << newline;
        json << indent << indent << "\"generator\": \"gpu_burn_monitoring\"," << newline;
        if (includeTimestamps_) {
            json << indent << indent << "\"timestamp\": \"" << formatTimestamp(snapshot.timestamp) << "\"," << newline;
            json << indent << indent << "\"timestamp_epoch\": " << toEpochMillis(snapshot.timestamp) << newline;
        }
        json << indent << "}," << newline;
        
        // Device info (if available from monitoring engine)
        json << indent << "\"device\": {" << newline;
        json << indent << indent << "\"id\": " << 0 << "," << newline;  // TODO: Get from engine
        json << indent << indent << "\"name\": \"" << "GPU" << "\"" << newline;  // TODO: Get actual name
        json << indent << "}," << newline;
        
        // Current metrics
        json << indent << "\"metrics\": {" << newline;
        
        // Health metrics
        json << indent << indent << "\"health\": {" << newline;
        json << indent << indent << indent << "\"temperature_celsius\": " << snapshot.health.temperature << "," << newline;
        json << indent << indent << indent << "\"power_watts\": " << snapshot.health.power << "," << newline;
        json << indent << indent << indent << "\"core_clock_mhz\": " << snapshot.health.coreClockMHz << "," << newline;
        json << indent << indent << indent << "\"memory_clock_mhz\": " << snapshot.health.memoryClockMHz << "," << newline;
        json << indent << indent << indent << "\"fan_speed_percent\": " << snapshot.health.fanSpeedPercent << newline;
        json << indent << indent << "}," << newline;
        
        // Performance metrics
        json << indent << indent << "\"performance\": {" << newline;
        json << indent << indent << indent << "\"current_gflops\": " << snapshot.performance.currentGFLOPS << "," << newline;
        json << indent << indent << indent << "\"average_gflops\": " << snapshot.performance.avgGFLOPS << "," << newline;
        json << indent << indent << indent << "\"peak_gflops\": " << snapshot.performance.peakGFLOPS << "," << newline;
        json << indent << indent << indent << "\"bandwidth_gbps\": " << snapshot.performance.bandwidthGBps << "," << newline;
        json << indent << indent << indent << "\"sm_efficiency_percent\": " << snapshot.performance.smEfficiency << newline;
        json << indent << indent << "}," << newline;
        
        // Error statistics
        json << indent << indent << "\"errors\": {" << newline;
        json << indent << indent << indent << "\"total_errors\": " << snapshot.errors.totalErrors << "," << newline;
        json << indent << indent << indent << "\"recent_errors\": " << snapshot.errors.recentErrors << "," << newline;
        json << indent << indent << indent << "\"errors_by_type\": {" << newline;
        
        bool first = true;
        for (const auto& [type, count] : snapshot.errors.errorsByType) {
            if (!first) json << "," << newline;
            json << indent << indent << indent << indent << "\"" << escapeString(type) << "\": " << count;
            first = false;
        }
        if (!snapshot.errors.errorsByType.empty()) json << newline;
        
        json << indent << indent << indent << "}" << newline;
        json << indent << indent << "}" << newline;
        json << indent << "}," << newline;
        
        // System status
        json << indent << "\"status\": {" << newline;
        json << indent << indent << "\"is_throttling\": " << (snapshot.status.isThrottling ? "true" : "false") << "," << newline;
        
        // Throttle reasons array
        json << indent << indent << "\"throttle_reasons\": [";
        if (!snapshot.status.throttleReasons.empty()) {
            json << newline;
            for (size_t i = 0; i < snapshot.status.throttleReasons.size(); ++i) {
                json << indent << indent << indent << "\"" << escapeString(snapshot.status.throttleReasons[i]) << "\"";
                if (i < snapshot.status.throttleReasons.size() - 1) json << ",";
                json << newline;
            }
            json << indent << indent;
        }
        json << "]," << newline;
        
        // Active alerts array
        json << indent << indent << "\"active_alerts\": [";
        if (!snapshot.status.activeAlerts.empty()) {
            json << newline;
            for (size_t i = 0; i < snapshot.status.activeAlerts.size(); ++i) {
                json << indent << indent << indent << "\"" << escapeString(snapshot.status.activeAlerts[i]) << "\"";
                if (i < snapshot.status.activeAlerts.size() - 1) json << ",";
                json << newline;
            }
            json << indent << indent;
        }
        json << "]" << newline;
        json << indent << "}";
        
        // Historical data (if available and enabled)
        if (includeHistorical_ && !historical.dataPoints.empty()) {
            json << "," << newline;
            json << indent << "\"historical\": {" << newline;
            json << indent << indent << "\"metric_name\": \"" << escapeString(historical.metricName) << "\"," << newline;
            json << indent << indent << "\"metric_type\": \"" << getMetricTypeName(historical.type) << "\"," << newline;
            
            // Statistics
            const_cast<TimeSeriesData&>(historical).calculateStats();
            json << indent << indent << "\"statistics\": {" << newline;
            json << indent << indent << indent << "\"min\": " << historical.stats.min << "," << newline;
            json << indent << indent << indent << "\"max\": " << historical.stats.max << "," << newline;
            json << indent << indent << indent << "\"average\": " << historical.stats.average << "," << newline;
            json << indent << indent << indent << "\"stddev\": " << historical.stats.stddev << "," << newline;
            json << indent << indent << indent << "\"count\": " << historical.stats.count << newline;
            json << indent << indent << "}," << newline;
            
            // Data points (limited to last 100 for space)
            json << indent << indent << "\"data_points\": [" << newline;
            size_t startIdx = historical.dataPoints.size() > 100 ? historical.dataPoints.size() - 100 : 0;
            for (size_t i = startIdx; i < historical.dataPoints.size(); ++i) {
                const auto& point = historical.dataPoints[i];
                json << indent << indent << indent << "{" << newline;
                json << indent << indent << indent << indent << "\"timestamp\": \"" << formatTimestamp(point.timestamp) << "\"," << newline;
                json << indent << indent << indent << indent << "\"value\": " << point.value << newline;
                json << indent << indent << indent << "}";
                if (i < historical.dataPoints.size() - 1) json << ",";
                json << newline;
            }
            json << indent << indent << "]" << newline;
            json << indent << "}";
        }
        
        json << newline << "}";
        
        return json.str();
    }
    
    // Helper functions
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    }
    
    long long toEpochMillis(const std::chrono::system_clock::time_point& tp) const {
        auto duration = tp.time_since_epoch();
        return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    }
    
    std::string escapeString(const std::string& str) const {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default:
                    if (c >= 0x20 && c <= 0x7E) {
                        escaped += c;
                    } else {
                        // Unicode escape
                        char buf[7];
                        snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                        escaped += buf;
                    }
            }
        }
        return escaped;
    }
    
private:
    bool prettyPrint_;
    bool includeTimestamps_;
    bool includeHistorical_;
};

// Factory function
REGISTER_REPORT_GENERATOR(JSONReporter)