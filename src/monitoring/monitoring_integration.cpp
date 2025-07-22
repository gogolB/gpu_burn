#include "monitoring_integration.h"
#include "metrics_collector.h"
#include "report_generator.h"
#include "data_store.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <filesystem>

MonitoringIntegration::MonitoringIntegration()
    : initialized_(false) {
}

MonitoringIntegration::~MonitoringIntegration() {
    if (initialized_) {
        stopMonitoring();
    }
}
bool MonitoringIntegration::initialize(const MonitoringIntegrationConfig& config) {
    config_ = config;
    
    // Create monitoring engine
    engine_ = std::make_unique<MonitoringEngine>();
    
    // Configure engine
    MonitoringConfig engineConfig;
    engineConfig.globalSamplingInterval = std::chrono::milliseconds(
        static_cast<int>(config.displayUpdateInterval * 1000));
    engineConfig.enableGPUHealth = config.collectGPUMetrics;
    engineConfig.enableSystemMetrics = config.collectSystemMetrics;
    engineConfig.enablePerformanceMetrics = config.collectPerformanceMetrics;
    engineConfig.enableErrorTracking = config.collectErrorMetrics;
    
    // Initialize engine
    if (!engine_->initialize(engineConfig)) {
        std::cerr << "Failed to initialize monitoring engine\n";
        return false;
    }
    
    // Register collectors with the engine
    registerCollectors();
    
    // Setup default callbacks
    setupDefaultCallbacks();
    setupEngineCallbacks();
    
    // Create report output directory if needed
    if (config.generateCSVReport || config.generateJSONReport) {
        std::filesystem::create_directories(config.reportOutputPath);
    }
    
    initialized_ = true;
    return true;
}

void MonitoringIntegration::startMonitoring() {
    if (!initialized_ || !engine_) {
        return;
    }
    
    engine_->startMonitoring();
}

void MonitoringIntegration::stopMonitoring() {
    if (!initialized_ || !engine_) {
        return;
    }
    
    engine_->stopMonitoring();
}

void MonitoringIntegration::registerCallbacks(const MonitoringCallbacks& callbacks) {
    callbacks_ = callbacks;
}

void MonitoringIntegration::updateKernelMetrics(const std::string& kernelName, 
                                               const KernelResult& result) {
    if (!engine_) return;
    
    // Record performance metrics
    if (result.success && engine_->getDataStore()) {
        auto* dataStore = engine_->getDataStore();
        dataStore->insert("kernel.execution_time_ms", result.executionTimeMs);
        dataStore->insert("kernel.gflops", result.gflops);
        dataStore->insert("kernel.memory_bandwidth_gbps", result.memoryBandwidthGBps);
        dataStore->insert("kernel.power_watts", result.avgPowerWatts);
        dataStore->insert("kernel.temperature_celsius", result.avgTemperatureCelsius);
        
        // Call performance callback if significant
        if (callbacks_.onPerformanceMilestone && result.gflops > 1000) {
            callbacks_.onPerformanceMilestone(result.gflops, result.memoryBandwidthGBps);
        }
    }
    
    // Call kernel complete callback
    if (callbacks_.onKernelComplete) {
        callbacks_.onKernelComplete(kernelName, result);
    }
}

void MonitoringIntegration::updateValidationMetrics(const ValidationResult& result) {
    if (!engine_ || !config_.trackValidationErrors || !engine_->getDataStore()) return;
    
    // Record validation metrics
    auto* dataStore = engine_->getDataStore();
    dataStore->insert("validation.checks_performed", 1.0);
    if (!result.passed) {
        dataStore->insert("validation.errors_detected", 1.0);
        dataStore->insert("validation.corrupted_elements", static_cast<double>(result.corruptedElements));
        
        // Call validation error callback
        if (callbacks_.onValidationError) {
            callbacks_.onValidationError(result);
        }
    }
    dataStore->insert("validation.overhead_ms", result.validationTimeMs);
}

void MonitoringIntegration::generateReport(const std::string& /* testName */) {
    if (!engine_) return;
    
    // Use the engine's report generation functionality
    if (config_.generateCSVReport) {
        engine_->generateReport("csv");
    }
    
    if (config_.generateJSONReport) {
        engine_->generateReport("json");
    }
}

void MonitoringIntegration::integrateWithKernel(KernelInterface* kernel) {
    if (!kernel || !engine_) return;
    
    kernel->setMonitoringEngine(engine_.get());
    kernel->enableMonitoring(true);
}

void MonitoringIntegration::integrateWithValidation(ValidationEngine* validationEngine) {
    if (!validationEngine || !engine_) return;
    
    // Setup validation tracking in monitoring
    config_.linkWithValidation = true;
    config_.trackValidationErrors = true;
}

void MonitoringIntegration::setupDefaultCallbacks() {
    // Default kernel start callback
    if (!callbacks_.onKernelStart) {
        callbacks_.onKernelStart = [this](const std::string& kernelName,
                                         const KernelConfig& /* config */) {
            if (config_.enableConsoleOutput) {
                std::cout << "[MONITOR] Starting kernel: " << kernelName << "\n";
            }
        };
    }
    
    // Default kernel complete callback
    if (!callbacks_.onKernelComplete) {
        callbacks_.onKernelComplete = [this](const std::string& kernelName,
                                           const KernelResult& result) {
            if (config_.enableConsoleOutput && !result.success) {
                std::cout << "[MONITOR] Kernel " << kernelName 
                         << " failed: " << result.errorMessage << "\n";
            }
        };
    }
    
    // Default system alert callback
    if (!callbacks_.onSystemAlert) {
        callbacks_.onSystemAlert = [this](const std::string& alertType,
                                        const std::string& message) {
            if (config_.enableConsoleOutput) {
                std::cout << "[ALERT] " << alertType << ": " << message << "\n";
            }
        };
    }
}

void MonitoringIntegration::setupEngineCallbacks() {
    if (!engine_) return;
    
    // Set alert callback
    engine_->setAlertCallback([this](const Alert& alert, double value) {
        if (!callbacks_.onSystemAlert) return;
        
        // Check thresholds based on alert type
        if (alert.metricName == "temperature" && value > config_.temperatureAlertThreshold) {
            std::stringstream ss;
            ss << "GPU temperature exceeded threshold: " << value << "°C";
            callbacks_.onSystemAlert("TEMPERATURE", ss.str());
        } else if (alert.metricName == "power" && value > config_.powerAlertThreshold) {
            std::stringstream ss;
            ss << "GPU power exceeded threshold: " << value << "W";
            callbacks_.onSystemAlert("POWER", ss.str());
        } else if (alert.metricName == "memory_usage" && value > config_.memoryUsageAlertThreshold) {
            std::stringstream ss;
            ss << "GPU memory usage exceeded threshold: " << value << "%";
            callbacks_.onSystemAlert("MEMORY", ss.str());
        }
    });
    
    // Register alerts with the engine
    Alert tempAlert;
    tempAlert.id = "temp_alert";
    tempAlert.name = "Temperature Alert";
    tempAlert.metricName = "temperature";
    tempAlert.threshold = config_.temperatureAlertThreshold;
    tempAlert.condition = [threshold = config_.temperatureAlertThreshold](double value) {
        return value > threshold;
    };
    engine_->registerAlert(tempAlert);
    
    Alert powerAlert;
    powerAlert.id = "power_alert";
    powerAlert.name = "Power Alert";
    powerAlert.metricName = "power";
    powerAlert.threshold = config_.powerAlertThreshold;
    powerAlert.condition = [threshold = config_.powerAlertThreshold](double value) {
        return value > threshold;
    };
    engine_->registerAlert(powerAlert);
    
    Alert memAlert;
    memAlert.id = "mem_alert";
    memAlert.name = "Memory Usage Alert";
    memAlert.metricName = "memory_usage";
    memAlert.threshold = config_.memoryUsageAlertThreshold;
    memAlert.condition = [threshold = config_.memoryUsageAlertThreshold](double value) {
        return value > threshold;
    };
    engine_->registerAlert(memAlert);
}

void MonitoringIntegration::registerCollectors() {
    if (!engine_) return;
    
    // Create factory functions for collectors
    extern std::unique_ptr<MetricsCollector> createPerformanceMonitor();
    extern std::unique_ptr<MetricsCollector> createGPUHealthMonitor();
    extern std::unique_ptr<MetricsCollector> createErrorMonitor();
    extern std::unique_ptr<MetricsCollector> createSystemMonitor();
    
    // Register performance monitor
    if (config_.collectPerformanceMetrics) {
        engine_->registerCollector(createPerformanceMonitor());
    }
    
    // Register GPU health monitor
    if (config_.collectGPUMetrics) {
        engine_->registerCollector(createGPUHealthMonitor());
    }
    
    // Register error monitor
    if (config_.collectErrorMetrics) {
        engine_->registerCollector(createErrorMonitor());
    }
    
    // Register system monitor
    if (config_.collectSystemMetrics) {
        engine_->registerCollector(createSystemMonitor());
    }
    
    // Register report generators
    registerReportGenerators();
}

void MonitoringIntegration::registerReportGenerators() {
    if (!engine_) return;
    
    // Create factory functions for report generators
    extern std::unique_ptr<ReportGenerator> createCSVReporter();
    extern std::unique_ptr<ReportGenerator> createJSONReporter();
    extern std::unique_ptr<ReportGenerator> createConsoleReporter();
    
    // Register CSV reporter
    if (config_.generateCSVReport) {
        engine_->registerReportGenerator(createCSVReporter());
    }
    
    // Register JSON reporter
    if (config_.generateJSONReport) {
        engine_->registerReportGenerator(createJSONReporter());
    }
    
    // Register console reporter for real-time display
    if (config_.enableRealTimeDisplay) {
        engine_->registerReportGenerator(createConsoleReporter());
    }
}

// MonitoringHelper namespace implementations
namespace MonitoringHelper {

std::unique_ptr<MonitoringIntegration> createGPUBurnMonitoring(
    const MonitoringIntegrationConfig& config) {
    
    auto integration = std::make_unique<MonitoringIntegration>();
    
    if (!integration->initialize(config)) {
        std::cerr << "Failed to initialize monitoring integration\n";
        return nullptr;
    }
    
    return integration;
}

MonitoringIntegrationConfig parseMonitoringArgs(int argc, char* argv[]) {
    MonitoringIntegrationConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--no-monitoring") {
            // This should be handled at a higher level
            config.enableRealTimeDisplay = false;
            config.enableConsoleOutput = false;
        } else if (arg == "--no-realtime") {
            config.enableRealTimeDisplay = false;
        } else if (arg == "--no-csv-report") {
            config.generateCSVReport = false;
        } else if (arg == "--no-json-report") {
            config.generateJSONReport = false;
        } else if (arg == "--report-path" && i + 1 < argc) {
            config.reportOutputPath = argv[++i];
        } else if (arg == "--display-interval" && i + 1 < argc) {
            config.displayUpdateInterval = std::stod(argv[++i]);
        }
    }
    
    return config;
}

void printMonitoringHelp() {
    std::cout << "\nMonitoring Options:\n";
    std::cout << "  --no-monitoring        Disable monitoring system\n";
    std::cout << "  --no-realtime          Disable real-time display\n";
    std::cout << "  --no-csv-report        Disable CSV report generation\n";
    std::cout << "  --no-json-report       Disable JSON report generation\n";
    std::cout << "  --report-path <path>   Output path for reports (default: ./reports/)\n";
    std::cout << "  --display-interval <s> Display update interval in seconds (default: 1.0)\n";
}

} // namespace MonitoringHelper