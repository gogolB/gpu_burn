#include "validation_engine.h"
#include "validation_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>

// CUDA kernel to check attention score stability
__global__ void check_attention_stability_kernel(
    const half* __restrict__ attention_scores,
    int* __restrict__ unstable_count,
    float* __restrict__ max_score,
    float* __restrict__ min_score,
    int num_elements,
    float threshold) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float score = __half2float(attention_scores[idx]);
        
        // Check for NaN or Inf
        if (!isfinite(score)) {
            atomicAdd(unstable_count, 1);
        }
        
        // Check for extreme values that might cause instability
        if (fabsf(score) > threshold) {
            atomicAdd(unstable_count, 1);
        }
        
        // Track min/max for analysis
        atomicMax((int*)max_score, __float_as_int(score));
        atomicMin((int*)min_score, __float_as_int(score));
    }
}

// CUDA kernel to detect gradient explosion/vanishing
__global__ void check_gradient_health_kernel(
    const half* __restrict__ gradients,
    int* __restrict__ explosion_count,
    int* __restrict__ vanishing_count,
    float* __restrict__ grad_norm,
    int num_elements,
    float explosion_threshold,
    float vanishing_threshold) {
    
    extern __shared__ float shared_norm[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Compute local gradient norm
    float local_sum = 0.0f;
    if (idx < num_elements) {
        float grad = __half2float(gradients[idx]);
        
        // Check for explosion
        if (fabsf(grad) > explosion_threshold) {
            atomicAdd(explosion_count, 1);
        }
        
        // Check for vanishing (but not zero)
        if (fabsf(grad) < vanishing_threshold && fabsf(grad) > 0.0f) {
            atomicAdd(vanishing_count, 1);
        }
        
        local_sum = grad * grad;
    }
    
    // Reduce to compute norm
    shared_norm[tid] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_norm[tid] += shared_norm[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd((unsigned long long*)grad_norm, 
                  __double_as_longlong((double)shared_norm[0]));
    }
}

// CUDA kernel to verify layer normalization numerical stability
__global__ void check_layernorm_stability_kernel(
    const half* __restrict__ normalized_output,
    const float* __restrict__ mean,
    const float* __restrict__ variance,
    int* __restrict__ unstable_count,
    int batch_size,
    int hidden_dim,
    float eps) {
    
    int batch_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx < batch_size && dim_idx < hidden_dim) {
        int idx = batch_idx * hidden_dim + dim_idx;
        float output = __half2float(normalized_output[idx]);
        
        // Check output is finite
        if (!isfinite(output)) {
            atomicAdd(unstable_count, 1);
            return;
        }
        
        // Verify mean is close to 0
        float batch_mean = mean[batch_idx];
        if (fabsf(batch_mean) > 1e-3f) {
            atomicAdd(unstable_count, 1);
        }
        
        // Verify variance is close to 1
        float batch_var = variance[batch_idx];
        if (fabsf(batch_var - 1.0f) > 1e-2f) {
            atomicAdd(unstable_count, 1);
        }
        
        // Check for numerical issues in variance computation
        if (batch_var < eps) {
            atomicAdd(unstable_count, 1);
        }
    }
}

// CUDA kernel to check mixed precision conversion accuracy
__global__ void check_mixed_precision_accuracy_kernel(
    const float* __restrict__ fp32_values,
    const half* __restrict__ fp16_values,
    const __nv_bfloat16* __restrict__ bf16_values,
    float* __restrict__ max_fp16_error,
    float* __restrict__ max_bf16_error,
    int* __restrict__ fp16_overflow_count,
    int* __restrict__ bf16_overflow_count,
    int num_elements) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float fp32_val = fp32_values[idx];
        
        // Check FP16 conversion
        float fp16_val = __half2float(fp16_values[idx]);
        float fp16_error = fabsf(fp32_val - fp16_val);
        
        // FP16 overflow check
        if (fabsf(fp32_val) > 65504.0f && isfinite(fp32_val)) {
            atomicAdd(fp16_overflow_count, 1);
        }
        
        atomicMax((int*)max_fp16_error, __float_as_int(fp16_error));
        
        // Check BF16 conversion
        float bf16_val = __bfloat162float(bf16_values[idx]);
        float bf16_error = fabsf(fp32_val - bf16_val);
        
        // BF16 has same exponent range as FP32, but check precision loss
        if (bf16_error > 0.01f * fabsf(fp32_val)) {  // More than 1% error
            atomicAdd(bf16_overflow_count, 1);
        }
        
        atomicMax((int*)max_bf16_error, __float_as_int(bf16_error));
    }
}

// Attention Score Stability Validator
class AttentionStabilityValidator : public ValidationMethod {
private:
    float stabilityThreshold_;
    int* d_unstableCount_;
    float* d_maxScore_;
    float* d_minScore_;
    
public:
    AttentionStabilityValidator(float threshold = 100.0f) 
        : stabilityThreshold_(threshold), d_unstableCount_(nullptr), 
          d_maxScore_(nullptr), d_minScore_(nullptr) {}
    
    ~AttentionStabilityValidator() {
        cleanup();
    }
    
    void setup(const KernelConfig& config) override {
        cudaMalloc(&d_unstableCount_, sizeof(int));
        cudaMalloc(&d_maxScore_, sizeof(float));
        cudaMalloc(&d_minScore_, sizeof(float));
        cudaMemset(d_unstableCount_, 0, sizeof(int));
        float min_init = FLT_MAX;
        float max_init = -FLT_MAX;
        cudaMemcpy(d_minScore_, &min_init, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_maxScore_, &max_init, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = getType();
        
        if (elementSize != sizeof(half)) {
            result.passed = false;
            result.errorDetails = "Attention validator expects FP16 data";
            return result;
        }
        
        // Reset counters
        cudaMemset(d_unstableCount_, 0, sizeof(int));
        
        // Check attention stability
        int threads = 256;
        int blocks = (numElements + threads - 1) / threads;
        check_attention_stability_kernel<<<blocks, threads>>>(
            static_cast<const half*>(data),
            d_unstableCount_,
            d_maxScore_,
            d_minScore_,
            numElements,
            stabilityThreshold_);
        
        cudaDeviceSynchronize();
        
        // Get results
        int unstableCount;
        float maxScore, minScore;
        cudaMemcpy(&unstableCount, d_unstableCount_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxScore, d_maxScore_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&minScore, d_minScore_, sizeof(float), cudaMemcpyDeviceToHost);
        
        result.passed = (unstableCount == 0);
        if (!result.passed) {
            result.errorDetails = "Attention scores unstable: " + std::to_string(unstableCount) + 
                                " unstable values detected. Range: [" + std::to_string(minScore) + 
                                ", " + std::to_string(maxScore) + "]";
            result.confidence = 1.0f - (float)unstableCount / numElements;
        } else {
            result.confidence = 1.0f;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Attention Score Stability Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::MATHEMATICAL_INVARIANT;  // Using existing type for LLM validation
    }
    
    void cleanup() override {
        if (d_unstableCount_) cudaFree(d_unstableCount_);
        if (d_maxScore_) cudaFree(d_maxScore_);
        if (d_minScore_) cudaFree(d_minScore_);
        d_unstableCount_ = nullptr;
        d_maxScore_ = nullptr;
        d_minScore_ = nullptr;
    }
};

// Gradient Health Validator
class GradientHealthValidator : public ValidationMethod {
private:
    float explosionThreshold_;
    float vanishingThreshold_;
    int* d_explosionCount_;
    int* d_vanishingCount_;
    float* d_gradNorm_;
    
public:
    GradientHealthValidator(float explosionThresh = 10.0f, float vanishingThresh = 1e-7f)
        : explosionThreshold_(explosionThresh), vanishingThreshold_(vanishingThresh),
          d_explosionCount_(nullptr), d_vanishingCount_(nullptr), d_gradNorm_(nullptr) {}
    
    ~GradientHealthValidator() {
        cleanup();
    }
    
    void setup(const KernelConfig& config) override {
        cudaMalloc(&d_explosionCount_, sizeof(int));
        cudaMalloc(&d_vanishingCount_, sizeof(int));
        cudaMalloc(&d_gradNorm_, sizeof(float));
        cudaMemset(d_explosionCount_, 0, sizeof(int));
        cudaMemset(d_vanishingCount_, 0, sizeof(int));
        cudaMemset(d_gradNorm_, 0, sizeof(float));
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = getType();
        
        if (elementSize != sizeof(half)) {
            result.passed = false;
            result.errorDetails = "Gradient validator expects FP16 data";
            return result;
        }
        
        // Reset counters
        cudaMemset(d_explosionCount_, 0, sizeof(int));
        cudaMemset(d_vanishingCount_, 0, sizeof(int));
        cudaMemset(d_gradNorm_, 0, sizeof(float));
        
        // Check gradient health
        int threads = 256;
        int blocks = (numElements + threads - 1) / threads;
        size_t sharedMem = threads * sizeof(float);
        
        check_gradient_health_kernel<<<blocks, threads, sharedMem>>>(
            static_cast<const half*>(data),
            d_explosionCount_,
            d_vanishingCount_,
            d_gradNorm_,
            numElements,
            explosionThreshold_,
            vanishingThreshold_);
        
        cudaDeviceSynchronize();
        
        // Get results
        int explosionCount, vanishingCount;
        float gradNorm;
        cudaMemcpy(&explosionCount, d_explosionCount_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&vanishingCount, d_vanishingCount_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&gradNorm, d_gradNorm_, sizeof(float), cudaMemcpyDeviceToHost);
        
        gradNorm = sqrtf(gradNorm / numElements);
        
        result.passed = (explosionCount == 0 && vanishingCount < numElements * 0.1f);
        if (!result.passed) {
            result.errorDetails = "Gradient health check failed: ";
            if (explosionCount > 0) {
                result.errorDetails += std::to_string(explosionCount) + " exploding gradients, ";
            }
            if (vanishingCount > numElements * 0.1f) {
                result.errorDetails += std::to_string(vanishingCount) + " vanishing gradients, ";
            }
            result.errorDetails += "Gradient norm: " + std::to_string(gradNorm);
            result.confidence = 1.0f - (float)(explosionCount + vanishingCount) / numElements;
        } else {
            result.confidence = 1.0f;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Gradient Health Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::MATHEMATICAL_INVARIANT;  // Using existing type for LLM validation
    }
    
    void cleanup() override {
        if (d_explosionCount_) cudaFree(d_explosionCount_);
        if (d_vanishingCount_) cudaFree(d_vanishingCount_);
        if (d_gradNorm_) cudaFree(d_gradNorm_);
        d_explosionCount_ = nullptr;
        d_vanishingCount_ = nullptr;
        d_gradNorm_ = nullptr;
    }
};

// Layer Normalization Stability Validator
class LayerNormStabilityValidator : public ValidationMethod {
private:
    int* d_unstableCount_;
    float eps_;
    
public:
    LayerNormStabilityValidator(float eps = 1e-5f)
        : d_unstableCount_(nullptr), eps_(eps) {}
    
    ~LayerNormStabilityValidator() {
        cleanup();
    }
    
    void setup(const KernelConfig& config) override {
        cudaMalloc(&d_unstableCount_, sizeof(int));
        cudaMemset(d_unstableCount_, 0, sizeof(int));
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = getType();
        
        // For layer norm, we need additional mean/variance data
        // In practice, these would be provided by the kernel
        // For now, we'll do a simpler stability check
        
        if (elementSize != sizeof(half)) {
            result.passed = false;
            result.errorDetails = "LayerNorm validator expects FP16 data";
            return result;
        }
        
        // Reset counter
        cudaMemset(d_unstableCount_, 0, sizeof(int));
        
        // Simple stability check - ensure no NaN/Inf values
        int threads = 256;
        int blocks = (numElements + threads - 1) / threads;
        
        // Reuse attention stability kernel for basic checks
        float *dummy_max, *dummy_min;
        cudaMalloc(&dummy_max, sizeof(float));
        cudaMalloc(&dummy_min, sizeof(float));
        
        check_attention_stability_kernel<<<blocks, threads>>>(
            static_cast<const half*>(data),
            d_unstableCount_,
            dummy_max,
            dummy_min,
            numElements,
            1000.0f);  // High threshold for layer norm output
        
        cudaDeviceSynchronize();
        
        // Get results
        int unstableCount;
        cudaMemcpy(&unstableCount, d_unstableCount_, sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(dummy_max);
        cudaFree(dummy_min);
        
        result.passed = (unstableCount == 0);
        if (!result.passed) {
            result.errorDetails = "LayerNorm output unstable: " + std::to_string(unstableCount) + 
                                " unstable values detected";
            result.confidence = 1.0f - (float)unstableCount / numElements;
        } else {
            result.confidence = 1.0f;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Layer Normalization Stability Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::MATHEMATICAL_INVARIANT;  // Using existing type for LLM validation
    }
    
    void cleanup() override {
        if (d_unstableCount_) cudaFree(d_unstableCount_);
        d_unstableCount_ = nullptr;
    }
};

// Mixed Precision Accuracy Validator
class MixedPrecisionAccuracyValidator : public ValidationMethod {
private:
    float* d_maxFP16Error_;
    float* d_maxBF16Error_;
    int* d_fp16OverflowCount_;
    int* d_bf16OverflowCount_;
    float errorThreshold_;
    
public:
    MixedPrecisionAccuracyValidator(float threshold = 0.01f)
        : d_maxFP16Error_(nullptr), d_maxBF16Error_(nullptr),
          d_fp16OverflowCount_(nullptr), d_bf16OverflowCount_(nullptr),
          errorThreshold_(threshold) {}
    
    ~MixedPrecisionAccuracyValidator() {
        cleanup();
    }
    
    void setup(const KernelConfig& config) override {
        cudaMalloc(&d_maxFP16Error_, sizeof(float));
        cudaMalloc(&d_maxBF16Error_, sizeof(float));
        cudaMalloc(&d_fp16OverflowCount_, sizeof(int));
        cudaMalloc(&d_bf16OverflowCount_, sizeof(int));
        
        float zero = 0.0f;
        cudaMemcpy(d_maxFP16Error_, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_maxBF16Error_, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_fp16OverflowCount_, 0, sizeof(int));
        cudaMemset(d_bf16OverflowCount_, 0, sizeof(int));
    }
    
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override {
        
        ValidationResult result;
        result.method = getType();
        
        // For mixed precision validation, we'd need reference FP32 data
        // Here we do a simplified check for range and stability
        
        if (elementSize != sizeof(half)) {
            result.passed = false;
            result.errorDetails = "Mixed precision validator expects FP16 data";
            return result;
        }
        
        // Check for values that would overflow in FP16
        const half* fp16_data = static_cast<const half*>(data);
        int overflowCount = 0;
        
        // Host-side check for demonstration (in practice, use GPU kernel)
        std::vector<half> h_data(numElements);
        cudaMemcpy(h_data.data(), fp16_data, numElements * sizeof(half), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < numElements; i++) {
            float val = __half2float(h_data[i]);
            if (!isfinite(val)) {
                overflowCount++;
            }
        }
        
        result.passed = (overflowCount == 0);
        if (!result.passed) {
            result.errorDetails = "Mixed precision overflow: " + std::to_string(overflowCount) + 
                                " values out of range";
            result.confidence = 1.0f - (float)overflowCount / numElements;
        } else {
            result.confidence = 1.0f;
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "Mixed Precision Accuracy Validator";
    }
    
    ValidationType getType() const override {
        return ValidationType::MATHEMATICAL_INVARIANT;  // Using existing type for LLM validation
    }
    
    void cleanup() override {
        if (d_maxFP16Error_) cudaFree(d_maxFP16Error_);
        if (d_maxBF16Error_) cudaFree(d_maxBF16Error_);
        if (d_fp16OverflowCount_) cudaFree(d_fp16OverflowCount_);
        if (d_bf16OverflowCount_) cudaFree(d_bf16OverflowCount_);
        d_maxFP16Error_ = nullptr;
        d_maxBF16Error_ = nullptr;
        d_fp16OverflowCount_ = nullptr;
        d_bf16OverflowCount_ = nullptr;
    }
};

// Factory functions for creating LLM validators
std::unique_ptr<ValidationMethod> createAttentionStabilityValidator(float threshold = 100.0f) {
    return std::make_unique<AttentionStabilityValidator>(threshold);
}

std::unique_ptr<ValidationMethod> createGradientHealthValidator(float explosionThresh = 10.0f, float vanishingThresh = 1e-7f) {
    return std::make_unique<GradientHealthValidator>(explosionThresh, vanishingThresh);
}

std::unique_ptr<ValidationMethod> createLayerNormStabilityValidator(float eps = 1e-5f) {
    return std::make_unique<LayerNormStabilityValidator>(eps);
}

std::unique_ptr<ValidationMethod> createMixedPrecisionAccuracyValidator(float threshold = 0.01f) {
    return std::make_unique<MixedPrecisionAccuracyValidator>(threshold);
}