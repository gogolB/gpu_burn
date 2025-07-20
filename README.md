# GPU Burn PyCUDA

A comprehensive GPU stress testing and validation framework designed to push NVIDIA GPUs to their limits while ensuring computational accuracy and hardware reliability.

## Features

- **Multiple Precision Support**
  - FP64 (Double Precision)
  - FP32 (Single Precision)
  - FP16 (Half Precision)
  - BF16 (Brain Float 16)
  - FP8 (8-bit Floating Point)
  - FP4 (4-bit Floating Point)
  - INT (Integer Operations)

- **Specialized Stress Kernels**
  - LLM Inference - Simulates large language model inference workloads
  - LLM Training - Replicates training patterns with gradient computations
  - Power Virus - Maximum power consumption patterns
  - Thermal Gradient - Creates temperature differentials across GPU
  - Memory Controller Stress - Targets memory subsystem bottlenecks
  - Mixed Precision Chaos - Simultaneous multi-precision operations

- **Comprehensive Validation Methods**
  - Checksum validation for data integrity
  - Golden reference comparison
  - Triple Modular Redundancy (TMR)
  - Spatial redundancy checking
  - Mathematical invariants verification

- **Real-time Monitoring and Reporting**
  - Console output with live metrics
  - CSV export for data analysis
  - JSON format for programmatic access
  - GPU health monitoring
  - Error detection and logging

## Requirements

- CUDA Toolkit 11.0 or later
- CMake 3.18 or higher
- C++17 compatible compiler
- NVIDIA GPU with compute capability 7.0+ (recommended)
  - Compute capability 6.0+ supported with limited features

## Building Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpu_burn_pycuda.git
cd gpu_burn_pycuda
```

2. Create build directory and configure:
```bash
mkdir build
cd build
cmake ..
```

3. Build the project:
```bash
make -j$(nproc)
```

## Usage Examples

### Basic stress test with FP32 precision:
```bash
./gpu_burn --precision fp32 --duration 300
```

### LLM inference workload with validation:
```bash
./gpu_burn --kernel llm_inference --validation checksum --duration 600
```

### Multi-GPU stress test with monitoring:
```bash
./gpu_burn --gpu all --precision fp16 --monitor --output results.json
```

### Power virus with thermal monitoring:
```bash
./gpu_burn --kernel power_virus --monitor-temp --threshold 85
```

## Available Kernels

- **fp64_kernel**: Double precision floating-point operations
- **fp32_kernel**: Single precision general matrix operations
- **fp16_kernel**: Half precision tensor core workloads
- **bf16_kernel**: Brain floating-point for AI workloads
- **fp8_kernel**: 8-bit floating-point operations
- **fp4_kernel**: Ultra-low precision stress testing
- **int_kernels**: Integer arithmetic stress patterns
- **llm_inference_kernel**: Simulates transformer inference patterns
- **llm_training_kernel**: Replicates training with backpropagation
- **power_virus_kernel**: Maximum power draw patterns
- **thermal_gradient_kernel**: Creates hot spots and thermal stress
- **memory_controller_stress_kernel**: Memory bandwidth saturation
- **mixed_precision_chaos_kernel**: Simultaneous multi-precision ops

## Validation Methods

- **Checksum Validation**: Fast integrity checking using GPU-accelerated checksums
- **Golden Reference**: Compares results against pre-computed reference values
- **Triple Modular Redundancy**: Executes kernels in triplicate and votes on results
- **Spatial Redundancy**: Distributes computation across multiple SMs for comparison
- **Mathematical Invariants**: Verifies results satisfy mathematical properties

## Monitoring and Reporting

The framework provides comprehensive monitoring capabilities:

- **Performance Metrics**: TFLOPS, bandwidth utilization, kernel execution times
- **Thermal Monitoring**: GPU temperature, hot spot detection, throttling events
- **Power Monitoring**: Power draw, efficiency metrics, power limit adherence
- **Error Detection**: Silent data corruption, ECC errors, kernel failures
- **Health Monitoring**: GPU utilization, memory usage, PCIe link status

Output formats:
- Console: Real-time display with color-coded alerts
- CSV: Timestamped metrics for analysis
- JSON: Structured data for integration with monitoring systems

## Project Structure

```
gpu_burn_pycuda/
├── CMakeLists.txt          # Build configuration
├── docs/                   # Detailed documentation
│   ├── advanced_workload_designs.md
│   ├── gpu_burn_upgrade_recommendations.md
│   └── monitoring_architecture.md
├── include/                # Header files
│   ├── kernel_interface.h
│   ├── validation_types.h
│   ├── monitoring_types.h
│   └── ...
├── src/
│   ├── host/              # Host-side code
│   ├── kernels/           # CUDA kernel implementations
│   ├── monitoring/        # Monitoring subsystem
│   └── validation/        # Validation implementations
└── tests/                 # Unit and integration tests
```

For detailed documentation, see the [docs/](docs/) directory.

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Follows the existing code style
- Includes appropriate tests
- Updates documentation as needed
- Passes all CI checks
---

For more information, bug reports, or feature requests, please visit our [GitHub repository](https://github.com/gogolb/gpu_burn).