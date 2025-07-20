#include "report_generator.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <vector>

// ANSI color codes for terminal output
namespace Color {
    const std::string RESET = "\033[0m";
    const std::string RED = "\033[31m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string BLUE = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN = "\033[36m";
    const std::string WHITE = "\033[37m";
    const std::string BOLD = "\033[1m";
    const std::string DIM = "\033[2m";
}

// Console Reporter - real-time console display with formatting
class ConsoleReporter : public ReportGenerator {
public:
    ConsoleReporter() {
        // Check if terminal supports color
        const char* term = std::getenv("TERM");
        supportsColor_ = (term != nullptr && std::string(term) != "dumb");
        
        // Configuration
        useSparklines_ = true;
        showTimestamps_ = true;
        compactMode_ = false;
    }
    
    void generate(const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        generate(std::cout, snapshot, historical);
    }
    
    void generate(std::ostream& output,
                 const MonitoringSnapshot& snapshot, 
                 const TimeSeriesData& historical) override {
        // Clear screen for refresh (optional)
        if (!compactMode_) {
            output << "\033[2J\033[H";  // Clear screen and move cursor to top
        }
        
        // Header
        printHeader(output, snapshot);
        
        // Health metrics
        printHealthSection(output, snapshot);
        
        // Performance metrics
        printPerformanceSection(output, snapshot);
        
        // Error statistics
        printErrorSection(output, snapshot);
        
        // System status
        printStatusSection(output, snapshot);
        
        // Historical trend (if available)
        if (!historical.dataPoints.empty()) {
            printHistoricalTrend(output, historical);
        }
        
        // Footer
        printFooter(output);
        
        output.flush();
    }
    
    bool generateToFile(const std::string& filename,
                       const MonitoringSnapshot& snapshot,
                       const TimeSeriesData& historical) override {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        // Disable colors for file output
        bool savedColorState = supportsColor_;
        supportsColor_ = false;
        
        generate(file, snapshot, historical);
        
        supportsColor_ = savedColorState;
        return true;
    }
    
    std::string getFormat() const override { return "console"; }
    
    void configure(const std::unordered_map<std::string, std::string>& options) override {
        auto it = options.find("color");
        if (it != options.end()) {
            supportsColor_ = (it->second == "true" || it->second == "1");
        }
        
        it = options.find("sparklines");
        if (it != options.end()) {
            useSparklines_ = (it->second == "true" || it->second == "1");
        }
        
        it = options.find("compact");
        if (it != options.end()) {
            compactMode_ = (it->second == "true" || it->second == "1");
        }
    }
    
private:
    void printHeader(std::ostream& out, const MonitoringSnapshot& snapshot) {
        out << color(Color::BOLD) << color(Color::CYAN);
        out << "╔══════════════════════════════════════════════════════════════════╗\n";
        out << "║                    GPU BURN MONITORING REPORT                    ║\n";
        out << "╚══════════════════════════════════════════════════════════════════╝\n";
        out << color(Color::RESET);
        
        if (showTimestamps_) {
            out << color(Color::DIM) << "Generated: " << formatTimestamp(snapshot.timestamp) 
                << color(Color::RESET) << "\n\n";
        }
    }
    
    void printHealthSection(std::ostream& out, const MonitoringSnapshot& snapshot) {
        out << color(Color::BOLD) << "🌡️  GPU Health Metrics\n" << color(Color::RESET);
        out << "├─────────────────────────────────────────────────────────────────\n";
        
        // Temperature with color coding
        double temp = snapshot.health.temperature;
        std::string tempColor = (temp > 80) ? Color::RED : 
                               (temp > 70) ? Color::YELLOW : Color::GREEN;
        
        out << "│ Temperature:     " << color(tempColor) 
            << std::fixed << std::setprecision(1) << temp << "°C" 
            << color(Color::RESET);
        
        if (useSparklines_ && temp > 0) {
            out << "  " << generateBar(temp, 100, 20);
        }
        out << "\n";
        
        // Power
        out << "│ Power Draw:      " << color(Color::CYAN)
            << std::fixed << std::setprecision(1) << snapshot.health.power << "W"
            << color(Color::RESET);
        
        if (useSparklines_ && snapshot.health.power > 0) {
            out << "   " << generateBar(snapshot.health.power, 400, 20);
        }
        out << "\n";
        
        // Clocks
        out << "│ Core Clock:      " << color(Color::BLUE)
            << std::fixed << std::setprecision(0) << snapshot.health.coreClockMHz << " MHz"
            << color(Color::RESET) << "\n";
        
        out << "│ Memory Clock:    " << color(Color::BLUE)
            << std::fixed << std::setprecision(0) << snapshot.health.memoryClockMHz << " MHz"
            << color(Color::RESET) << "\n";
        
        // Fan speed
        out << "│ Fan Speed:       " << color(Color::GREEN)
            << std::fixed << std::setprecision(0) << snapshot.health.fanSpeedPercent << "%"
            << color(Color::RESET) << "\n";
        
        out << "└─────────────────────────────────────────────────────────────────\n\n";
    }
    
    void printPerformanceSection(std::ostream& out, const MonitoringSnapshot& snapshot) {
        out << color(Color::BOLD) << "⚡ Performance Metrics\n" << color(Color::RESET);
        out << "├─────────────────────────────────────────────────────────────────\n";
        
        // GFLOPS
        out << "│ Current GFLOPS:  " << color(Color::MAGENTA)
            << std::fixed << std::setprecision(2) << snapshot.performance.currentGFLOPS
            << color(Color::RESET) << "\n";
        
        out << "│ Average GFLOPS:  " << color(Color::MAGENTA)
            << std::fixed << std::setprecision(2) << snapshot.performance.avgGFLOPS
            << color(Color::RESET) << "\n";
        
        out << "│ Peak GFLOPS:     " << color(Color::BOLD) << color(Color::MAGENTA)
            << std::fixed << std::setprecision(2) << snapshot.performance.peakGFLOPS
            << color(Color::RESET) << "\n";
        
        // Bandwidth
        out << "│ Bandwidth:       " << color(Color::CYAN)
            << std::fixed << std::setprecision(2) << snapshot.performance.bandwidthGBps << " GB/s"
            << color(Color::RESET) << "\n";
        
        // SM Efficiency
        out << "│ SM Efficiency:   " << color(Color::GREEN)
            << std::fixed << std::setprecision(1) << snapshot.performance.smEfficiency << "%"
            << color(Color::RESET);
        
        if (useSparklines_ && snapshot.performance.smEfficiency > 0) {
            out << "  " << generateBar(snapshot.performance.smEfficiency, 100, 20);
        }
        out << "\n";
        
        out << "└─────────────────────────────────────────────────────────────────\n\n";
    }
    
    void printErrorSection(std::ostream& out, const MonitoringSnapshot& snapshot) {
        out << color(Color::BOLD) << "⚠️  Error Statistics\n" << color(Color::RESET);
        out << "├─────────────────────────────────────────────────────────────────\n";
        
        bool hasErrors = snapshot.errors.totalErrors > 0;
        
        // Total errors with color coding
        std::string errorColor = hasErrors ? Color::RED : Color::GREEN;
        out << "│ Total Errors:    " << color(errorColor)
            << snapshot.errors.totalErrors
            << color(Color::RESET) << "\n";
        
        // Recent errors
        out << "│ Recent Errors:   " << color(errorColor)
            << snapshot.errors.recentErrors << " (last minute)"
            << color(Color::RESET) << "\n";
        
        // Error breakdown
        if (!snapshot.errors.errorsByType.empty()) {
            out << "│ Error Types:\n";
            for (const auto& [type, count] : snapshot.errors.errorsByType) {
                out << "│   - " << std::setw(20) << std::left << type 
                    << ": " << color(Color::YELLOW) << count << color(Color::RESET) << "\n";
            }
        }
        
        out << "└─────────────────────────────────────────────────────────────────\n\n";
    }
    
    void printStatusSection(std::ostream& out, const MonitoringSnapshot& snapshot) {
        if (!snapshot.status.isThrottling && snapshot.status.activeAlerts.empty()) {
            return;  // Skip if no issues
        }
        
        out << color(Color::BOLD) << "🚨 System Status\n" << color(Color::RESET);
        out << "├─────────────────────────────────────────────────────────────────\n";
        
        // Throttling
        if (snapshot.status.isThrottling) {
            out << "│ " << color(Color::RED) << color(Color::BOLD) 
                << "⚠️  THROTTLING DETECTED" << color(Color::RESET) << "\n";
            
            if (!snapshot.status.throttleReasons.empty()) {
                out << "│ Reasons:\n";
                for (const auto& reason : snapshot.status.throttleReasons) {
                    out << "│   • " << reason << "\n";
                }
            }
        }
        
        // Active alerts
        if (!snapshot.status.activeAlerts.empty()) {
            out << "│ Active Alerts:\n";
            for (const auto& alert : snapshot.status.activeAlerts) {
                out << "│   ⚠️  " << color(Color::YELLOW) << alert << color(Color::RESET) << "\n";
            }
        }
        
        out << "└─────────────────────────────────────────────────────────────────\n\n";
    }
    
    void printHistoricalTrend(std::ostream& out, const TimeSeriesData& historical) {
        if (!useSparklines_) return;
        
        out << color(Color::BOLD) << "📈 Historical Trend\n" << color(Color::RESET);
        out << "├─────────────────────────────────────────────────────────────────\n";
        
        // Generate sparkline for the data
        std::string sparkline = generateSparkline(historical.dataPoints, 50);
        out << "│ " << historical.metricName << ": " << sparkline << "\n";
        
        // Stats
        const_cast<TimeSeriesData&>(historical).calculateStats();
        out << "│ Min: " << std::fixed << std::setprecision(2) << historical.stats.min
            << ", Max: " << historical.stats.max
            << ", Avg: " << historical.stats.average << "\n";
        
        out << "└─────────────────────────────────────────────────────────────────\n\n";
    }
    
    void printFooter(std::ostream& out) {
        out << color(Color::DIM) 
            << "─────────────────────────────────────────────────────────────────\n"
            << "Press Ctrl+C to stop monitoring\n"
            << color(Color::RESET);
    }
    
    // Helper functions
    std::string color(const std::string& code) const {
        return supportsColor_ ? code : "";
    }
    
    std::string generateBar(double value, double maxValue, int width) const {
        int filled = static_cast<int>((value / maxValue) * width);
        filled = std::max(0, std::min(width, filled));
        
        std::string bar = "[";
        for (int i = 0; i < width; ++i) {
            if (i < filled) {
                bar += "█";
            } else {
                bar += "░";
            }
        }
        bar += "]";
        
        return bar;
    }
    
    std::string generateSparkline(const std::vector<TimeSeriesDataPoint<double>>& data, int width) const {
        if (data.empty()) return "";
        
        const std::string sparks = "▁▂▃▄▅▆▇█";
        
        // Find min/max
        double min = data[0].value;
        double max = data[0].value;
        for (const auto& point : data) {
            min = std::min(min, point.value);
            max = std::max(max, point.value);
        }
        
        // Sample data points
        std::string sparkline;
        int step = std::max(1, static_cast<int>(data.size() / width));
        
        for (size_t i = 0; i < data.size(); i += step) {
            double normalized = (max > min) ? (data[i].value - min) / (max - min) : 0.0;
            int index = static_cast<int>(normalized * (sparks.length() - 1));
            sparkline += sparks[index];
        }
        
        return sparkline;
    }
    
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const {
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
private:
    bool supportsColor_;
    bool useSparklines_;
    bool showTimestamps_;
    bool compactMode_;
};

// Factory function
REGISTER_REPORT_GENERATOR(ConsoleReporter)