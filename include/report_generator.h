#ifndef REPORT_GENERATOR_H
#define REPORT_GENERATOR_H

#include <string>
#include <ostream>
#include <memory>
#include "monitoring_types.h"

// Forward declaration
class MonitoringEngine;

// Abstract interface for report generation
class ReportGenerator {
public:
    virtual ~ReportGenerator() = default;
    
    // Generate report with current snapshot and optional historical data
    virtual void generate(const MonitoringSnapshot& snapshot, 
                         const TimeSeriesData& historical) = 0;
    
    // Generate report to output stream
    virtual void generate(std::ostream& output,
                         const MonitoringSnapshot& snapshot, 
                         const TimeSeriesData& historical) = 0;
    
    // Generate report to file
    virtual bool generateToFile(const std::string& filename,
                               const MonitoringSnapshot& snapshot,
                               const TimeSeriesData& historical) = 0;
    
    // Get format name
    virtual std::string getFormat() const = 0;
    
    // Configuration
    virtual void configure(const std::unordered_map<std::string, std::string>& options) {}
    
    // Set monitoring engine reference
    void setMonitoringEngine(MonitoringEngine* engine) { engine_ = engine; }
    MonitoringEngine* getMonitoringEngine() const { return engine_; }
    
protected:
    MonitoringEngine* engine_ = nullptr;
    
    // Helper methods for derived classes
    std::string formatTimestamp(const std::chrono::system_clock::time_point& tp) const;
    std::string formatDuration(const std::chrono::duration<double>& duration) const;
    std::string formatBytes(size_t bytes) const;
    std::string formatPercentage(double value) const;
    std::string formatMetricValue(double value, const std::string& unit = "") const;
};

// Factory function type for creating report generators
using ReportGeneratorFactory = std::unique_ptr<ReportGenerator>(*)();

// Helper macro for report generator registration
#define REGISTER_REPORT_GENERATOR(GeneratorClass) \
    std::unique_ptr<ReportGenerator> create##GeneratorClass() { \
        return std::make_unique<GeneratorClass>(); \
    }

#endif // REPORT_GENERATOR_H