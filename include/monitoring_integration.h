#ifndef MONITORING_INTEGRATION_H
#define MONITORING_INTEGRATION_H

#include <memory>
#include <string>
#include <functional>
#include "monitoring_engine.h"
#include "kernel_interface.h"
#include "validation_types.h"

// Monitoring integration configuration
struct MonitoringIntegrationConfig {
    // Display options
    bool enableRealTimeDisplay = true;
    bool enableConsoleOutput = true;
    double displayUpdateInterval = 1.0;  // seconds
    
    // Report generation
    bool generateCSVReport = true;
    bool generateJSONReport = true;
    std::string reportOutputPath = "./reports/";
    
    // Metric collection
    bool collectGPUMetrics = true;
    bool collectSystemMetrics = true;
    bool collectPerformanceMetrics = true;
    bool collectErrorMetrics = true;
    
    // Alert thresholds
    double temperatureAlertThreshold = 85.0;  // Celsius
    double powerAlertThreshold = 550.0;       // Watts
    double memoryUsageAlertThreshold = 95.0;  // Percentage
    
    // Integration with validation
    bool linkWithValidation = true;
    bool trackValidationErrors = true;
};

// Monitoring callbacks for kernel execution
struct MonitoringCallbacks {
    // Called before kernel execution
    std::function<void(const std::string& kernelName, const KernelConfig& config)> onKernelStart;
    
    // Called after kernel execution
    std::function<void(const std::string& kernelName, const KernelResult& result)> onKernelComplete;
    
    // Called on validation error
    std::function<void(const ValidationResult& result)> onValidationError;
    
    // Called on performance milestone
    std::function<void(double gflops, double bandwidth)> onPerformanceMilestone;
    
    // Called on system alert
    std::function<void(const std::string& alertType, const std::string& message)> onSystemAlert;
};

// Monitoring integration manager
class MonitoringIntegration {
public:
    MonitoringIntegration();
    ~MonitoringIntegration();
    
    // Initialize monitoring with configuration
    bool initialize(const MonitoringIntegrationConfig& config);
    
    // Start monitoring
    void startMonitoring();
    
    // Stop monitoring
    void stopMonitoring();
    
    // Register callbacks
    void registerCallbacks(const MonitoringCallbacks& callbacks);
    
    // Update metrics during kernel execution
    void updateKernelMetrics(const std::string& kernelName, const KernelResult& result);
    
    // Update validation metrics
    void updateValidationMetrics(const ValidationResult& result);
    
    // Generate final report
    void generateReport(const std::string& testName);
    
    // Get monitoring engine instance
    MonitoringEngine* getEngine() { return engine_.get(); }
    
    // Helper methods for common integrations
    void integrateWithKernel(KernelInterface* kernel);
    void integrateWithValidation(class ValidationEngine* validationEngine);
    
private:
    std::unique_ptr<MonitoringEngine> engine_;
    MonitoringIntegrationConfig config_;
    MonitoringCallbacks callbacks_;
    bool initialized_;
    
    // Setup monitoring callbacks
    void setupDefaultCallbacks();
    void setupEngineCallbacks();
    
    // Register collectors with the engine
    void registerCollectors();
    
    // Register report generators with the engine
    void registerReportGenerators();
};

// Global monitoring integration helper functions
namespace MonitoringHelper {
    // Create and configure monitoring for GPU burn tests
    std::unique_ptr<MonitoringIntegration> createGPUBurnMonitoring(
        const MonitoringIntegrationConfig& config = MonitoringIntegrationConfig());
    
    // Parse command line arguments for monitoring configuration
    MonitoringIntegrationConfig parseMonitoringArgs(int argc, char* argv[]);
    
    // Print monitoring help
    void printMonitoringHelp();
}

#endif // MONITORING_INTEGRATION_H