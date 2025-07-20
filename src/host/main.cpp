#include <iostream>
#include <vector>
#include <memory>
#include <map>
#include <iomanip>
#include <string>
#include <cstring>
#include <chrono>
#include <functional>

#include "kernel_interface.h"
#include "gpu_utils.h"
#include "validation_engine.h"
#include "golden_reference.h"
#include "checksum_validator.h"
#include "mathematical_invariants.h"
#include "tmr_validator.h"
#include "spatial_redundancy.h"
#include "monitoring_integration.h"

// Kernel factory declarations
std::unique_ptr<KernelInterface> createFP32Kernel();
std::unique_ptr<KernelInterface> createFP64Kernel();
std::unique_ptr<KernelInterface> createFP16Kernel();
std::unique_ptr<KernelInterface> createBF16Kernel();
std::unique_ptr<KernelInterface> createFP8Kernel();
std::unique_ptr<KernelInterface> createFP4Kernel();
std::unique_ptr<KernelInterface> createUINT8Kernel();
std::unique_ptr<KernelInterface> createUINT16Kernel();

// New stress kernel factory declarations
std::unique_ptr<KernelInterface> createPowerVirusKernel();
std::unique_ptr<KernelInterface> createThermalGradientKernel();
std::unique_ptr<KernelInterface> createMemoryControllerStressKernel();
std::unique_ptr<KernelInterface> createMixedPrecisionChaosKernel();

// LLM workload kernel factory declarations
std::unique_ptr<KernelInterface> createLLMInferenceKernel();
std::unique_ptr<KernelInterface> createLLMTrainingKernel();

// Command line options
struct Options {
    int deviceId = 0;
    size_t matrixSize = 1024;
    size_t iterations = 100;
    bool useTensorCores = true;
    std::vector<std::string> precisionTypes;
    bool listGpus = false;
    bool showHelp = false;
    int duration = 60; // Run duration in seconds
    
    // Validation options
    std::vector<std::string> validationMethods;
    bool enableValidation = false;
    bool injectSDC = false;
    std::string sdcType = "bitflip";
    double sdcRate = 0.0001;
    double validationInterval = 1.0;
    bool continuousValidation = false;
    
    // Monitoring options
    bool enableMonitoring = true;
    bool realTimeDisplay = true;
    bool generateCSVReport = true;
    bool generateJSONReport = true;
    std::string reportPath = "./reports/";
    double displayUpdateInterval = 1.0;
};

void printHelp(const char* programName) {
    std::cout << "GPU Burn - Comprehensive GPU Stress Test with SDC Detection\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -d, --device <id>      GPU device ID (default: 0)\n";
    std::cout << "  -s, --size <size>      Matrix size (default: 1024)\n";
    std::cout << "  -i, --iterations <num> Iterations per kernel (default: 100)\n";
    std::cout << "  -t, --duration <sec>   Test duration in seconds (default: 60)\n";
    std::cout << "  -p, --precision <type> Precision types to test (can specify multiple)\n";
    std::cout << "                         Available: fp32, fp64, fp16, bf16, fp8, fp4, uint8, uint16\n";
    std::cout << "                         Default: all supported types\n";
    std::cout << "  --no-tensor-cores      Disable tensor cores\n";
    std::cout << "  -l, --list             List available GPUs\n";
    std::cout << "  -h, --help             Show this help message\n\n";
    std::cout << "Validation Options:\n";
    std::cout << "  --validate <method>    Enable validation (can specify multiple)\n";
    std::cout << "                         Methods: golden, checksum, invariant, tmr, spatial, all\n";
    std::cout << "  --inject-sdc <type>    Inject SDC errors for testing\n";
    std::cout << "                         Types: bitflip, memory, timing, thermal\n";
    std::cout << "  --sdc-rate <prob>      SDC injection probability (default: 0.0001)\n";
    std::cout << "  --validation-interval  Validation interval in seconds (default: 1.0)\n";
    std::cout << "  --continuous-validation Enable continuous validation mode\n\n";
    std::cout << "Monitoring Options:\n";
    std::cout << "  --no-monitoring        Disable monitoring system\n";
    std::cout << "  --no-realtime          Disable real-time display\n";
    std::cout << "  --no-csv-report        Disable CSV report generation\n";
    std::cout << "  --no-json-report       Disable JSON report generation\n";
    std::cout << "  --report-path <path>   Output path for reports (default: ./reports/)\n";
    std::cout << "  --display-interval <s> Display update interval in seconds (default: 1.0)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " -d 0 -s 2048 -p fp32 fp16\n";
    std::cout << "  " << programName << " --duration 300 --no-tensor-cores\n";
    std::cout << "  " << programName << " --validate golden checksum --inject-sdc bitflip\n";
    std::cout << "  " << programName << " --validate all --sdc-rate 0.001\n";
}

Options parseArgs(int argc, char* argv[]) {
    Options opts;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            opts.showHelp = true;
            return opts;
        } else if (arg == "-l" || arg == "--list") {
            opts.listGpus = true;
            return opts;
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                opts.deviceId = std::stoi(argv[++i]);
            }
        } else if (arg == "-s" || arg == "--size") {
            if (i + 1 < argc) {
                opts.matrixSize = std::stoul(argv[++i]);
            }
        } else if (arg == "-i" || arg == "--iterations") {
            if (i + 1 < argc) {
                opts.iterations = std::stoul(argv[++i]);
            }
        } else if (arg == "-t" || arg == "--duration") {
            if (i + 1 < argc) {
                opts.duration = std::stoi(argv[++i]);
            }
        } else if (arg == "-p" || arg == "--precision") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                opts.precisionTypes.push_back(argv[++i]);
            }
        } else if (arg == "--no-tensor-cores") {
            opts.useTensorCores = false;
        } else if (arg == "--validate") {
            opts.enableValidation = true;
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string method = argv[++i];
                if (method == "all") {
                    opts.validationMethods = {"golden", "checksum", "invariant", "tmr", "spatial"};
                } else {
                    opts.validationMethods.push_back(method);
                }
            }
        } else if (arg == "--inject-sdc") {
            if (i + 1 < argc) {
                opts.injectSDC = true;
                opts.sdcType = argv[++i];
            }
        } else if (arg == "--sdc-rate") {
            if (i + 1 < argc) {
                opts.sdcRate = std::stod(argv[++i]);
            }
        } else if (arg == "--validation-interval") {
            if (i + 1 < argc) {
                opts.validationInterval = std::stod(argv[++i]);
            }
        } else if (arg == "--continuous-validation") {
            opts.continuousValidation = true;
        } else if (arg == "--no-monitoring") {
            opts.enableMonitoring = false;
        } else if (arg == "--no-realtime") {
            opts.realTimeDisplay = false;
        } else if (arg == "--no-csv-report") {
            opts.generateCSVReport = false;
        } else if (arg == "--no-json-report") {
            opts.generateJSONReport = false;
        } else if (arg == "--report-path") {
            if (i + 1 < argc) {
                opts.reportPath = argv[++i];
            }
        } else if (arg == "--display-interval") {
            if (i + 1 < argc) {
                opts.displayUpdateInterval = std::stod(argv[++i]);
            }
        }
    }
    
    return opts;
}

void listGpus() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    std::cout << "Available GPUs:\n\n";
    
    for (int i = 0; i < deviceCount; i++) {
        GpuInfo info = getGpuInfo(i);
        GpuArchitecture arch = detectArchitecture(info);
        
        std::cout << "Device " << i << ": " << info.name << "\n";
        std::cout << "  Architecture: " << architectureToString(arch) 
                  << " (SM " << info.computeCapabilityMajor << "." 
                  << info.computeCapabilityMinor << ")\n";
        std::cout << "  Memory: " << (info.totalMemory / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB\n";
        std::cout << "  Multiprocessors: " << info.multiProcessorCount << "\n";
        std::cout << "  Max Threads/Block: " << info.maxThreadsPerBlock << "\n";
        std::cout << "  Tensor Cores: " << (info.supportsTensorCores() ? "Yes" : "No") << "\n";
        std::cout << "  FP16 Support: " << (info.supportsFP16() ? "Yes" : "No") << "\n";
        std::cout << "  BF16 Support: " << (info.supportsBF16() ? "Yes" : "No") << "\n";
        std::cout << "  FP8 Support: " << (info.supportsFP8() ? "Yes" : "No") << "\n";
        std::cout << "\n";
    }
}

void runKernel(KernelInterface* kernel, const KernelConfig& config,
               int duration, const std::string& precision,
               ValidationEngine* validationEngine,
               MonitoringIntegration* monitoring) {
    std::cout << "\n=== Testing " << precision << " ===\n";
    std::cout << "Kernel: " << kernel->getName() << "\n";
    std::cout << "Description: " << kernel->getDescription() << "\n";
    
    if (!kernel->isSupported(config.deviceId)) {
        std::cout << "Not supported on this GPU\n";
        return;
    }
    
    size_t memReq = kernel->getMemoryRequirement(config);
    std::cout << "Memory requirement: " << (memReq / (1024.0 * 1024.0)) << " MB\n";
    
    // Set validation engine if enabled
    if (validationEngine && config.enableValidation) {
        kernel->setValidationEngine(validationEngine);
        kernel->enableValidation(true);
        std::cout << "Validation: Enabled\n";
    }
    
    // Set monitoring if enabled
    if (monitoring && monitoring->getEngine()) {
        kernel->setMonitoringEngine(monitoring->getEngine());
        kernel->enableMonitoring(true);
        std::cout << "Monitoring: Enabled\n";
    }
    
    // Warmup
    std::cout << "Warming up...\n";
    KernelConfig warmupConfig = config;
    warmupConfig.numIterations = 10;
    warmupConfig.enableValidation = false;  // Don't validate during warmup
    kernel->execute(warmupConfig);
    
    // Run for specified duration
    std::cout << "Running for " << duration << " seconds...\n";
    
    auto startTime = std::chrono::high_resolution_clock::now();
    double totalGflops = 0;
    double totalBandwidth = 0;
    int runs = 0;
    
    // Validation statistics
    size_t totalSDCDetected = 0;
    size_t totalSDCCorrected = 0;
    double totalValidationTime = 0;
    
    while (true) {
        KernelResult result = kernel->execute(config);
        
        if (!result.success) {
            std::cerr << "Error: " << result.errorMessage << "\n";
            break;
        }
        
        totalGflops += result.gflops;
        totalBandwidth += result.memoryBandwidthGBps;
        runs++;
        
        // Update validation statistics
        if (result.validationPerformed) {
            totalSDCDetected += result.sdcDetectedCount;
            totalSDCCorrected += result.sdcCorrectedCount;
            totalValidationTime += result.validationOverheadMs;
            
            // Print validation errors if found
            for (const auto& valResult : result.validationResults) {
                if (!valResult.passed) {
                    std::cout << "\n[VALIDATION ERROR] " << valResult.errorDetails << "\n";
                }
            }
            
            // Update monitoring with validation results
            if (monitoring) {
                for (const auto& valResult : result.validationResults) {
                    monitoring->updateValidationMetrics(valResult);
                }
            }
        }
        
        // Update monitoring with kernel results
        if (monitoring) {
            monitoring->updateKernelMetrics(precision, result);
        }
        
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            currentTime - startTime).count();
        
        if (elapsed >= duration) {
            break;
        }
        
        // Print progress
        if (runs % 10 == 0) {
            std::cout << "\rProgress: " << elapsed << "/" << duration << "s, "
                     << "Avg GFLOPS: " << std::fixed << std::setprecision(2)
                     << (totalGflops / runs);
            if (config.enableValidation) {
                std::cout << ", SDCs: " << totalSDCDetected;
            }
            std::cout << std::flush;
        }
    }
    
    std::cout << "\n\nResults:\n";
    std::cout << "  Total runs: " << runs << "\n";
    std::cout << "  Average GFLOPS: " << std::fixed << std::setprecision(2)
              << (totalGflops / runs) << "\n";
    std::cout << "  Average Memory Bandwidth: " << std::fixed << std::setprecision(2)
              << (totalBandwidth / runs) << " GB/s\n";
    
    if (config.enableValidation && runs > 0) {
        std::cout << "\nValidation Results:\n";
        std::cout << "  Total SDCs detected: " << totalSDCDetected << "\n";
        std::cout << "  Total SDCs corrected: " << totalSDCCorrected << "\n";
        std::cout << "  Average validation overhead: " << std::fixed << std::setprecision(2)
                  << (totalValidationTime / runs) << " ms\n";
        
        if (validationEngine) {
            ValidationStats stats = validationEngine->getStats();
            std::cout << "  Total validation checks: " << stats.totalChecks << "\n";
            std::cout << "  Failed checks: " << stats.failedChecks << "\n";
            if (stats.injectedErrors > 0) {
                std::cout << "  SDC detection rate: " << std::fixed << std::setprecision(1)
                          << (stats.detectionRate * 100) << "%\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        Options opts = parseArgs(argc, argv);
        
        if (opts.showHelp) {
            printHelp(argv[0]);
            return 0;
        }
        
        if (opts.listGpus) {
            listGpus();
            return 0;
        }
        
        // Print GPU information
        std::cout << "GPU Burn - Comprehensive GPU Stress Test\n";
        std::cout << "========================================\n\n";
        
        GpuInfo info = getGpuInfo(opts.deviceId);
        GpuArchitecture arch = detectArchitecture(info);
        
        std::cout << "Using GPU: " << info.name << "\n";
        std::cout << "Architecture: " << architectureToString(arch) << "\n";
        std::cout << "Compute Capability: " << info.computeCapabilityMajor << "." 
                  << info.computeCapabilityMinor << "\n";
        std::cout << "Memory: " << (info.totalMemory / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB\n";
        std::cout << "Tensor Cores: " << (info.supportsTensorCores() ? "Enabled" : "Disabled") 
                  << "\n\n";
        
        // Create validation engine if enabled
        std::unique_ptr<ValidationEngine> validationEngine;
        if (opts.enableValidation) {
            validationEngine = std::make_unique<ValidationEngine>();
            
            // Configure validation
            ValidationConfig valConfig;
            valConfig.validationInterval = opts.validationInterval;
            valConfig.continuousValidation = opts.continuousValidation;
            
            // Configure SDC injection
            if (opts.injectSDC) {
                if (opts.sdcType == "bitflip") {
                    valConfig.injectionType = SDCInjectionType::BITFLIP;
                } else if (opts.sdcType == "memory") {
                    valConfig.injectionType = SDCInjectionType::MEMORY_PATTERN;
                } else if (opts.sdcType == "timing") {
                    valConfig.injectionType = SDCInjectionType::TIMING;
                } else if (opts.sdcType == "thermal") {
                    valConfig.injectionType = SDCInjectionType::THERMAL;
                }
                valConfig.sdcProbability = opts.sdcRate;
            }
            
            validationEngine->initialize(valConfig);
            
            // Register validation methods
            for (const auto& method : opts.validationMethods) {
                if (method == "golden") {
                    validationEngine->registerMethod(std::make_unique<GoldenReferenceValidator>());
                } else if (method == "checksum") {
                    validationEngine->registerMethod(std::make_unique<ChecksumValidator>());
                } else if (method == "invariant") {
                    auto invariantValidator = std::make_unique<MathematicalInvariantsValidator>();
                    invariantValidator->addInvariant(InvariantType::MATRIX_TRACE, 0.0, 1e-5);
                    validationEngine->registerMethod(std::move(invariantValidator));
                } else if (method == "tmr") {
                    validationEngine->registerMethod(std::make_unique<TMRValidator>());
                } else if (method == "spatial") {
                    validationEngine->registerMethod(std::make_unique<SpatialRedundancyValidator>());
                }
            }
            
            // Enable validation types
            valConfig.validationTypes = ValidationType::ALL;
            validationEngine->initialize(valConfig);
        }
        
        // Create kernel configuration
        KernelConfig config;
        config.deviceId = opts.deviceId;
        config.matrixSize = opts.matrixSize;
        config.numIterations = opts.iterations;
        config.useTensorCores = opts.useTensorCores && info.supportsTensorCores();
        config.gridDim = dim3(256, 1, 1);
        config.blockDim = dim3(256, 1, 1);
        config.enableValidation = opts.enableValidation;
        config.injectSDC = opts.injectSDC;
        config.sdcProbability = opts.sdcRate;
        
        std::cout << "Test Configuration:\n";
        std::cout << "  Matrix Size: " << config.matrixSize << " x " << config.matrixSize << "\n";
        std::cout << "  Iterations per run: " << config.numIterations << "\n";
        std::cout << "  Duration: " << opts.duration << " seconds\n";
        std::cout << "  Tensor Cores: " << (config.useTensorCores ? "Enabled" : "Disabled") << "\n";
        
        if (opts.enableValidation) {
            std::cout << "\nValidation Configuration:\n";
            std::cout << "  Validation methods: ";
            for (const auto& method : opts.validationMethods) {
                std::cout << method << " ";
            }
            std::cout << "\n";
            if (opts.injectSDC) {
                std::cout << "  SDC injection: " << opts.sdcType
                          << " (rate: " << opts.sdcRate << ")\n";
            }
        }
        
        // Create kernel registry
        std::map<std::string, std::function<std::unique_ptr<KernelInterface>()>> kernelRegistry = {
            {"fp32", createFP32Kernel},
            {"fp64", createFP64Kernel},
            {"fp16", createFP16Kernel},
            {"bf16", createBF16Kernel},
            {"fp8", createFP8Kernel},
            {"fp4", createFP4Kernel},
            {"uint8", createUINT8Kernel},
            {"uint16", createUINT16Kernel},
            // New stress kernels
            {"power_virus", createPowerVirusKernel},
            {"thermal_gradient", createThermalGradientKernel},
            {"memory_stress", createMemoryControllerStressKernel},
            {"mixed_chaos", createMixedPrecisionChaosKernel},
            // LLM workload kernels
            {"llm_inference", createLLMInferenceKernel},
            {"llm_training", createLLMTrainingKernel}
        };
        
        // Determine which kernels to run
        std::vector<std::string> kernelsToRun;
        if (opts.precisionTypes.empty()) {
            // Run all supported kernels
            for (const auto& [name, factory] : kernelRegistry) {
                auto kernel = factory();
                if (kernel->isSupported(opts.deviceId)) {
                    kernelsToRun.push_back(name);
                }
            }
        } else {
            kernelsToRun = opts.precisionTypes;
        }
        
        std::cout << "\nKernels to run: ";
        for (const auto& name : kernelsToRun) {
            std::cout << name << " ";
        }
        std::cout << "\n";
        
        // Run each kernel
        for (const auto& kernelName : kernelsToRun) {
            auto it = kernelRegistry.find(kernelName);
            if (it != kernelRegistry.end()) {
                auto kernel = it->second();
                
                // Create monitoring integration if enabled
                std::unique_ptr<MonitoringIntegration> monitoring;
                if (opts.enableMonitoring) {
                    MonitoringIntegrationConfig monitorConfig;
                    monitorConfig.enableRealTimeDisplay = opts.realTimeDisplay;
                    monitorConfig.enableConsoleOutput = opts.realTimeDisplay;
                    monitorConfig.generateCSVReport = opts.generateCSVReport;
                    monitorConfig.generateJSONReport = opts.generateJSONReport;
                    monitorConfig.reportOutputPath = opts.reportPath;
                    monitorConfig.displayUpdateInterval = opts.displayUpdateInterval;
                    monitorConfig.linkWithValidation = opts.enableValidation;
                    
                    monitoring = MonitoringHelper::createGPUBurnMonitoring(monitorConfig);
                    if (monitoring) {
                        monitoring->startMonitoring();
                        
                        // Integrate with validation if enabled
                        if (validationEngine) {
                            monitoring->integrateWithValidation(validationEngine.get());
                        }
                    }
                }
                
                runKernel(kernel.get(), config, opts.duration, kernelName,
                         validationEngine.get(), monitoring.get());
                
                // Stop monitoring and generate report
                if (monitoring) {
                    monitoring->stopMonitoring();
                    monitoring->generateReport("gpu_burn_" + kernelName);
                }
            } else {
                std::cerr << "Unknown kernel type: " << kernelName << "\n";
            }
        }
        
        // Print final validation statistics if enabled
        if (opts.enableValidation && validationEngine) {
            std::cout << "\n\n=== Final Validation Statistics ===\n";
            ValidationStats finalStats = validationEngine->getStats();
            std::cout << "Total validation checks: " << finalStats.totalChecks << "\n";
            std::cout << "Failed checks: " << finalStats.failedChecks << "\n";
            std::cout << "Injected errors: " << finalStats.injectedErrors << "\n";
            std::cout << "Detected errors: " << finalStats.detectedErrors << "\n";
            if (finalStats.injectedErrors > 0) {
                std::cout << "Detection rate: " << std::fixed << std::setprecision(1)
                          << (finalStats.detectionRate * 100) << "%\n";
            }
            
            // Print failure breakdown by method
            if (!finalStats.failuresByMethod.empty()) {
                std::cout << "\nFailures by validation method:\n";
                for (const auto& [type, count] : finalStats.failuresByMethod) {
                    std::string methodName;
                    switch (type) {
                        case ValidationType::GOLDEN_REFERENCE: methodName = "Golden Reference"; break;
                        case ValidationType::CHECKSUM: methodName = "Checksum"; break;
                        case ValidationType::MATHEMATICAL_INVARIANT: methodName = "Mathematical Invariant"; break;
                        case ValidationType::TMR: methodName = "TMR"; break;
                        case ValidationType::SPATIAL_REDUNDANCY: methodName = "Spatial Redundancy"; break;
                        default: methodName = "Unknown"; break;
                    }
                    std::cout << "  " << methodName << ": " << count << "\n";
                }
            }
        }
        
        // Generate final monitoring report if enabled
        if (opts.enableMonitoring) {
            std::cout << "\n\n=== Monitoring Summary ===\n";
            std::cout << "Reports generated in: " << opts.reportPath << "\n";
            if (opts.generateCSVReport) {
                std::cout << "  CSV reports: gpu_burn_*.csv\n";
            }
            if (opts.generateJSONReport) {
                std::cout << "  JSON reports: gpu_burn_*.json\n";
            }
        }
        
        std::cout << "\n\nGPU Burn test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}