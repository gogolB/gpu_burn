#include "validation_engine.h"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Temporal Redundancy Validator - Validates results across time windows
// Designed for dynamic workloads like thermal gradients and power patterns

class TemporalRedundancyValidator : public ValidationMethod {
private:
    struct TemporalSnapshot {
        std::vector<float> data;
        double timestamp;
        size_t iteration;
    };
    
    std::vector<TemporalSnapshot> snapshots_;
    size_t max_snapshots_ = 10;
    double tolerance_ = 1e-3;
    
    // Device buffers for temporal comparison
    float* d_reference_ = nullptr;
    float* d_diff_buffer_ = nullptr;
    size_t buffer_size_ = 0;
    
public:
    ~TemporalRedundancyValidator() {
        cleanup();
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = ValidationType::TEMPORAL_REDUNDANCY;
        result.passed = true;
        
        if (elementSize != sizeof(float)) {
            result.passed = false;
            result.errorDetails = "Temporal redundancy only supports float data";
            return result;
        }
        
        const float* float_data = static_cast<const float*>(data);
        
        // Allocate device buffers if needed
        if (buffer_size_ < numElements) {
            cleanup();
            CUDA_CHECK(cudaMalloc(&d_reference_, numElements * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_diff_buffer_, numElements * sizeof(float)));
            buffer_size_ = numElements;
        }
        
        // Create current snapshot
        TemporalSnapshot current;
        current.data.resize(numElements);
        current.timestamp = getCurrentTime();
        current.iteration = config.numIterations;
        
        // Copy data to host
        CUDA_CHECK(cudaMemcpy(current.data.data(), float_data, 
                             numElements * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Validate against previous snapshots
        if (!snapshots_.empty()) {
            // Check for temporal consistency
            validateTemporalProgression(current, result, numElements);
            
            // Check for unexpected variations
            validateTemporalStability(current, result, numElements);
            
            // Check for power/thermal patterns
            validateTemporalPatterns(current, result, numElements);
        }
        
        // Store snapshot
        snapshots_.push_back(current);
        if (snapshots_.size() > max_snapshots_) {
            snapshots_.erase(snapshots_.begin());
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Temporal Redundancy Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::TEMPORAL_REDUNDANCY;
    }
    
    void setup(const KernelConfig& config) override {
        // Clear previous snapshots
        snapshots_.clear();
        
        // Adjust tolerance based on workload type
        if (config.matrixSize > 2048) {
            tolerance_ = 1e-2; // Relax for larger problems
        }
    }
    
    void cleanup() override {
        if (d_reference_) {
            cudaFree(d_reference_);
            d_reference_ = nullptr;
        }
        if (d_diff_buffer_) {
            cudaFree(d_diff_buffer_);
            d_diff_buffer_ = nullptr;
        }
        buffer_size_ = 0;
        snapshots_.clear();
    }
    
private:
    double getCurrentTime() {
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event);
        cudaEventSynchronize(event);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, event, event);
        cudaEventDestroy(event);
        return double(milliseconds);
    }
    
    void validateTemporalProgression(const TemporalSnapshot& current,
                                    ValidationResult& result,
                                    size_t numElements) {
        // Check if values are progressing as expected
        const auto& prev = snapshots_.back();
        
        double max_change = 0.0;
        double avg_change = 0.0;
        size_t anomalies = 0;
        
        for (size_t i = 0; i < numElements; i++) {
            double change = std::abs(current.data[i] - prev.data[i]);
            max_change = std::max(max_change, change);
            avg_change += change;
            
            // Check for unexpected jumps
            if (change > 10.0 * tolerance_ && std::abs(prev.data[i]) > tolerance_) {
                anomalies++;
            }
        }
        
        avg_change /= numElements;
        
        if (anomalies > numElements * 0.01) { // More than 1% anomalies
            result.passed = false;
            result.errorDetails = "Temporal progression anomalies detected: " +
                                std::to_string(anomalies) + " elements with unexpected changes";
            result.corruptedElements = anomalies;
        }
    }
    
    void validateTemporalStability(const TemporalSnapshot& current,
                                  ValidationResult& result,
                                  size_t numElements) {
        // Check variance across time windows
        if (snapshots_.size() < 3) return;
        
        std::vector<double> variances;
        
        // Calculate variance for each element across snapshots
        for (size_t i = 0; i < numElements; i += 1000) { // Sample every 1000th element
            double mean = 0.0;
            for (const auto& snap : snapshots_) {
                mean += snap.data[i];
            }
            mean /= snapshots_.size();
            
            double variance = 0.0;
            for (const auto& snap : snapshots_) {
                double diff = snap.data[i] - mean;
                variance += diff * diff;
            }
            variance /= snapshots_.size();
            variances.push_back(variance);
        }
        
        // Check for excessive variance
        double max_variance = *std::max_element(variances.begin(), variances.end());
        double avg_variance = std::accumulate(variances.begin(), variances.end(), 0.0) / variances.size();
        
        if (max_variance > 100.0 * tolerance_) {
            result.passed = false;
            result.errorDetails = "Temporal instability detected: excessive variance = " +
                                std::to_string(max_variance);
        }
    }
    
    void validateTemporalPatterns(const TemporalSnapshot& current,
                                ValidationResult& result,
                                size_t numElements) {
        // Look for patterns that indicate power/thermal issues
        if (snapshots_.size() < 5) return;
        
        // Check for oscillations (potential VRM issues)
        size_t oscillations = 0;
        for (size_t i = 0; i < numElements; i += 100) {
            bool increasing = false;
            int direction_changes = 0;
            
            for (size_t s = 1; s < snapshots_.size(); s++) {
                bool current_increasing = snapshots_[s].data[i] > snapshots_[s-1].data[i];
                if (s > 1 && current_increasing != increasing) {
                    direction_changes++;
                }
                increasing = current_increasing;
            }
            
            if (direction_changes > snapshots_.size() / 2) {
                oscillations++;
            }
        }
        
        if (oscillations > numElements / 1000) {
            // Add to error details as a warning
            if (!result.errorDetails.empty()) {
                result.errorDetails += "; ";
            }
            result.errorDetails += "Warning: Temporal oscillation pattern detected - possible power delivery issue";
        }
    }
};
// CUDA kernel functions must be defined outside of class scope
__global__ void computeSMChecksumsKernel(const float* data, float* sm_results,
                                        size_t elements_per_sm, int num_sms) {
    // Approximate SM ID (device-specific)
    int sm_id = blockIdx.x % num_sms;
    int local_tid = threadIdx.x;
    int threads_per_block = blockDim.x;
    
    __shared__ float shared_sum[256];
    shared_sum[local_tid] = 0.0f;
    
    // Each SM processes its portion of data
    size_t start_idx = sm_id * elements_per_sm;
    size_t end_idx = (start_idx + elements_per_sm < elements_per_sm * num_sms) ?
                     start_idx + elements_per_sm : elements_per_sm * num_sms;
    
    // Compute checksum for this SM's data
    float local_sum = 0.0f;
    for (size_t i = start_idx + local_tid; i < end_idx; i += threads_per_block) {
        float val = data[i];
        // Simple checksum with position weighting
        local_sum += val * (1.0f + float(i % 1000) * 0.001f);
    }
    
    shared_sum[local_tid] = local_sum;
    __syncthreads();
    
    // Reduction within block
    for (int stride = threads_per_block / 2; stride > 0; stride /= 2) {
        if (local_tid < stride) {
            shared_sum[local_tid] += shared_sum[local_tid + stride];
        }
        __syncthreads();
    }
    
    // Store result
    if (local_tid == 0) {
        sm_results[blockIdx.x] = shared_sum[0];
    }
}

// Cross-SM Validator - Validates consistency across streaming multiprocessors
class CrossSMValidator : public ValidationMethod {
private:
    int num_sms_;
    float* d_sm_results_ = nullptr;
    size_t sm_buffer_size_ = 0;
    
public:
    ~CrossSMValidator() {
        cleanup();
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = ValidationType::CROSS_SM;
        result.passed = true;
        
        if (elementSize != sizeof(float)) {
            result.passed = false;
            result.errorDetails = "Cross-SM validation only supports float data";
            return result;
        }
        
        // Get device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, config.deviceId));
        num_sms_ = prop.multiProcessorCount;
        
        const float* float_data = static_cast<const float*>(data);
        
        // Allocate SM results buffer
        size_t results_per_sm = numElements / num_sms_;
        if (sm_buffer_size_ < num_sms_ * results_per_sm) {
            cleanup();
            sm_buffer_size_ = num_sms_ * results_per_sm;
            CUDA_CHECK(cudaMalloc(&d_sm_results_, sm_buffer_size_ * sizeof(float)));
        }
        
        // Launch kernel to compute per-SM checksums
        computeSMChecksums(float_data, numElements, config);
        
        // Validate cross-SM consistency
        validateSMConsistency(result, results_per_sm);
        
        return result;
    }
    
    std::string getName() const override {
        return "Cross-SM Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::CROSS_SM;
    }
    
    void setup(const KernelConfig& config) override {
        // Get device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, config.deviceId);
        num_sms_ = prop.multiProcessorCount;
    }
    
    void cleanup() override {
        if (d_sm_results_) {
            cudaFree(d_sm_results_);
            d_sm_results_ = nullptr;
        }
        sm_buffer_size_ = 0;
    }
    
private:
    void computeSMChecksums(const float* data, size_t numElements, const KernelConfig& config) {
        size_t elements_per_sm = numElements / num_sms_;
        int blocks_per_sm = 4;
        int total_blocks = num_sms_ * blocks_per_sm;
        
        computeSMChecksumsKernel<<<total_blocks, 256>>>(
            data, d_sm_results_, elements_per_sm, num_sms_);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    void validateSMConsistency(ValidationResult& result, size_t results_per_sm) {
        // Copy SM results to host
        std::vector<float> h_sm_results(num_sms_ * 4);
        CUDA_CHECK(cudaMemcpy(h_sm_results.data(), d_sm_results_, 
                             h_sm_results.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Group results by SM
        std::vector<float> sm_averages(num_sms_, 0.0f);
        for (int sm = 0; sm < num_sms_; sm++) {
            for (int b = 0; b < 4; b++) {
                sm_averages[sm] += h_sm_results[sm * 4 + b];
            }
            sm_averages[sm] /= 4.0f;
        }
        
        // Check for outliers
        float mean = 0.0f;
        for (float avg : sm_averages) {
            mean += avg;
        }
        mean /= num_sms_;
        
        float variance = 0.0f;
        for (float avg : sm_averages) {
            float diff = avg - mean;
            variance += diff * diff;
        }
        variance /= num_sms_;
        float stddev = std::sqrt(variance);
        
        // Detect faulty SMs
        int faulty_sms = 0;
        for (int sm = 0; sm < num_sms_; sm++) {
            if (std::abs(sm_averages[sm] - mean) > 3.0f * stddev) {
                faulty_sms++;
                if (result.errorDetails.empty()) {
                    result.errorDetails = "Faulty SM detected: SM" + std::to_string(sm) +
                                        " deviation = " + std::to_string(std::abs(sm_averages[sm] - mean) / stddev) + " sigma";
                }
            }
        }
        
        if (faulty_sms > 0) {
            result.passed = false;
            result.corruptedElements = faulty_sms;
        }
    }
};

// Power-Aware Validator - Validates during power state transitions
class PowerAwareValidator : public ValidationMethod {
private:
    enum PowerState {
        IDLE,
        RAMP_UP,
        PEAK,
        RAMP_DOWN
    };
    
    PowerState current_state_ = IDLE;
    size_t state_counter_ = 0;
    std::vector<float> baseline_values_;
    float power_tolerance_multiplier_ = 1.0f;
    
public:
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = ValidationType::POWER_AWARE;
        result.passed = true;
        
        if (elementSize != sizeof(float)) {
            result.passed = false;
            result.errorDetails = "Power-aware validation only supports float data";
            return result;
        }
        
        const float* float_data = static_cast<const float*>(data);
        
        // Update power state
        updatePowerState(config);
        
        // Adjust validation based on power state
        switch (current_state_) {
            case IDLE:
                validateIdleState(float_data, numElements, result);
                break;
            case RAMP_UP:
                validateRampUpState(float_data, numElements, result);
                break;
            case PEAK:
                validatePeakState(float_data, numElements, result);
                break;
            case RAMP_DOWN:
                validateRampDownState(float_data, numElements, result);
                break;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Power-Aware Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::POWER_AWARE;
    }
    
    void setup(const KernelConfig& config) override {
        current_state_ = IDLE;
        state_counter_ = 0;
        baseline_values_.clear();
    }
    
    void cleanup() override {
        baseline_values_.clear();
    }
    
private:
    void updatePowerState(const KernelConfig& config) {
        state_counter_++;
        
        // Simple state machine based on iteration count
        size_t cycle_length = 40;
        size_t position = state_counter_ % cycle_length;
        
        if (position < 5) {
            current_state_ = IDLE;
            power_tolerance_multiplier_ = 1.0f;
        } else if (position < 15) {
            current_state_ = RAMP_UP;
            power_tolerance_multiplier_ = 2.0f; // More tolerance during transitions
        } else if (position < 30) {
            current_state_ = PEAK;
            power_tolerance_multiplier_ = 1.5f;
        } else {
            current_state_ = RAMP_DOWN;
            power_tolerance_multiplier_ = 2.0f;
        }
    }
    
    void validateIdleState(const float* data, size_t numElements, ValidationResult& result) {
        // Store baseline during idle
        if (baseline_values_.empty()) {
            baseline_values_.resize(std::min(size_t(1000), numElements));
            CUDA_CHECK(cudaMemcpy(baseline_values_.data(), data, 
                                 baseline_values_.size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    
    void validateRampUpState(const float* data, size_t numElements, ValidationResult& result) {
        // Check for power-related computation errors during ramp
        std::vector<float> sample(100);
        size_t stride = numElements / 100;
        
        for (size_t i = 0; i < 100; i++) {
            CUDA_CHECK(cudaMemcpy(&sample[i], &data[i * stride], sizeof(float), cudaMemcpyDeviceToHost));
        }
        
        // Look for NaN/Inf that might indicate power issues
        for (float val : sample) {
            if (std::isnan(val) || std::isinf(val)) {
                result.passed = false;
                result.errorDetails = "Invalid values detected during power ramp-up";
                result.corruptedElements++;
            }
        }
    }
    
    void validatePeakState(const float* data, size_t numElements, ValidationResult& result) {
        // More stringent validation during peak power
        if (baseline_values_.empty()) return;
        
        std::vector<float> current_values(baseline_values_.size());
        CUDA_CHECK(cudaMemcpy(current_values.data(), data, 
                             current_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Check for excessive deviations that might indicate power-induced errors
        size_t deviations = 0;
        for (size_t i = 0; i < baseline_values_.size(); i++) {
            float expected_range = std::abs(baseline_values_[i]) * 10.0f * power_tolerance_multiplier_;
            if (std::abs(current_values[i] - baseline_values_[i]) > expected_range) {
                deviations++;
            }
        }
        
        if (deviations > baseline_values_.size() * 0.05) { // More than 5% deviations
            result.passed = false;
            result.errorDetails = "Excessive deviations during peak power: " +
                                std::to_string(deviations) + " out of " +
                                std::to_string(baseline_values_.size());
            result.corruptedElements = deviations;
        }
    }
    
    void validateRampDownState(const float* data, size_t numElements, ValidationResult& result) {
        // Check for stability during power reduction
        validateRampUpState(data, numElements, result); // Similar checks
    }
};

// Register new validation methods
void registerAdvancedValidators(ValidationEngine* engine) {
    engine->registerMethod(std::make_unique<TemporalRedundancyValidator>());
    engine->registerMethod(std::make_unique<CrossSMValidator>());
    engine->registerMethod(std::make_unique<PowerAwareValidator>());
}