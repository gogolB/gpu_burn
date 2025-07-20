#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

// Helper functions for activation functions
__device__ __forceinline__ float gelu(float x) {
    // Approximation of GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float c = 0.797884560802865f; // sqrt(2/pi)
    const float a = 0.044715f;
    float x3 = x * x * x;
    float tanh_arg = c * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(tanh_arg));
}

__device__ __forceinline__ float silu(float x) {
    // SiLU (Swish): x * sigmoid(x)
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ half gelu_half(half x) {
    float xf = __half2float(x);
    return __float2half(gelu(xf));
}

__device__ __forceinline__ half silu_half(half x) {
    float xf = __half2float(x);
    return __float2half(silu(xf));
}

// Attention kernel with flash attention pattern for memory efficiency
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int seq_len,
    int num_heads,
    float scale) {
    
    extern __shared__ half shared_mem[];
    half* s_q = shared_mem;
    half* s_k = shared_mem + BLOCK_SIZE * HEAD_DIM;
    half* s_v = s_k + BLOCK_SIZE * HEAD_DIM;
    float* s_scores = (float*)(s_v + BLOCK_SIZE * HEAD_DIM);
    
    int head_idx = blockIdx.y;
    int q_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    if (q_idx >= seq_len) return;
    
    // Load query tile into shared memory
    for (int d = 0; d < HEAD_DIM; d++) {
        if (tid < BLOCK_SIZE && q_idx < seq_len) {
            s_q[tid * HEAD_DIM + d] = Q[(head_idx * seq_len + q_idx) * HEAD_DIM + d];
        }
    }
    __syncthreads();
    
    // Initialize output accumulator
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }
    
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    
    // Process KV blocks
    for (int kv_block = 0; kv_block < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; kv_block++) {
        int kv_idx = kv_block * BLOCK_SIZE + tid;
        
        // Load K and V tiles
        for (int d = 0; d < HEAD_DIM; d++) {
            if (kv_idx < seq_len) {
                s_k[tid * HEAD_DIM + d] = K[(head_idx * seq_len + kv_idx) * HEAD_DIM + d];
                s_v[tid * HEAD_DIM + d] = V[(head_idx * seq_len + kv_idx) * HEAD_DIM + d];
            }
        }
        __syncthreads();
        
        // Compute attention scores for this block
        if (tid < BLOCK_SIZE && q_idx < seq_len) {
            float score = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                score += __half2float(s_q[tid * HEAD_DIM + d]) * 
                        __half2float(s_k[tid * HEAD_DIM + d]);
            }
            score *= scale;
            
            // Online softmax update
            float old_max = row_max;
            row_max = fmaxf(row_max, score);
            float exp_score = expf(score - row_max);
            float exp_old_max = expf(old_max - row_max);
            
            row_sum = row_sum * exp_old_max + exp_score;
            
            // Update accumulator with proper scaling
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] = acc[d] * exp_old_max + 
                        exp_score * __half2float(s_v[tid * HEAD_DIM + d]);
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (tid < BLOCK_SIZE && q_idx < seq_len) {
        for (int d = 0; d < HEAD_DIM; d++) {
            O[(head_idx * seq_len + q_idx) * HEAD_DIM + d] = 
                __float2half(acc[d] / row_sum);
        }
    }
}

// KV-cache update kernel simulating inference memory patterns
__global__ void kv_cache_update_kernel(
    const half* __restrict__ new_k,
    const half* __restrict__ new_v,
    half* __restrict__ cache_k,
    half* __restrict__ cache_v,
    int* __restrict__ cache_pos,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_heads * head_dim;
    
    if (idx < total_elements) {
        int batch_idx = idx / (num_heads * head_dim);
        int pos = cache_pos[batch_idx];
        
        if (pos < max_seq_len) {
            int offset = batch_idx * max_seq_len * num_heads * head_dim + 
                        pos * num_heads * head_dim + 
                        (idx % (num_heads * head_dim));
            
            cache_k[offset] = new_k[idx];
            cache_v[offset] = new_v[idx];
            
            // Simulate cache line access patterns
            if (threadIdx.x % 32 == 0) {
                atomicAdd(cache_pos + batch_idx, 0); // Force memory barrier
            }
        }
    }
}

// Mixed precision GEMM kernel for Q, K, V projections
template<int TILE_SIZE>
__global__ void mixed_precision_qkv_projection(
    const half* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight_q,
    const __nv_bfloat16* __restrict__ weight_k,
    const __nv_bfloat16* __restrict__ weight_v,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int seq_len,
    int hidden_dim,
    int proj_dim) {
    
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_weight[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    // Compute Q, K, V projections with mixed precision
    float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
    
    for (int t = 0; t < (hidden_dim + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load input tile (FP16 to FP32)
        if (row < seq_len && t * TILE_SIZE + tx < hidden_dim) {
            tile_input[ty][tx] = __half2float(input[row * hidden_dim + t * TILE_SIZE + tx]);
        } else {
            tile_input[ty][tx] = 0.0f;
        }
        
        // Load weight tiles (BF16 to FP32) and compute
        if (col < proj_dim && t * TILE_SIZE + ty < hidden_dim) {
            // Q projection
            tile_weight[ty][tx] = __bfloat162float(weight_q[(t * TILE_SIZE + ty) * proj_dim + col]);
            __syncthreads();
            for (int k = 0; k < TILE_SIZE; k++) {
                sum_q += tile_input[ty][k] * tile_weight[k][tx];
            }
            
            // K projection
            tile_weight[ty][tx] = __bfloat162float(weight_k[(t * TILE_SIZE + ty) * proj_dim + col]);
            __syncthreads();
            for (int k = 0; k < TILE_SIZE; k++) {
                sum_k += tile_input[ty][k] * tile_weight[k][tx];
            }
            
            // V projection
            tile_weight[ty][tx] = __bfloat162float(weight_v[(t * TILE_SIZE + ty) * proj_dim + col]);
            __syncthreads();
            for (int k = 0; k < TILE_SIZE; k++) {
                sum_v += tile_input[ty][k] * tile_weight[k][tx];
            }
        }
        __syncthreads();
    }
    
    // Write outputs (FP32 to FP16)
    if (row < seq_len && col < proj_dim) {
        q_out[row * proj_dim + col] = __float2half(sum_q);
        k_out[row * proj_dim + col] = __float2half(sum_k);
        v_out[row * proj_dim + col] = __float2half(sum_v);
    }
}

// Positional encoding computation kernel
__global__ void rotary_positional_encoding_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    const float* __restrict__ freqs,
    int seq_len,
    int num_heads,
    int head_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = seq_len * num_heads * head_dim / 2;
    
    if (idx < total_elements) {
        int pos = idx / (num_heads * head_dim / 2);
        int head_idx = (idx / (head_dim / 2)) % num_heads;
        int dim_idx = idx % (head_dim / 2);
        
        float freq = freqs[dim_idx];
        float angle = pos * freq;
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        
        int base_idx = pos * num_heads * head_dim + head_idx * head_dim + 2 * dim_idx;
        
        // Apply rotation to Q
        float q0 = __half2float(q[base_idx]);
        float q1 = __half2float(q[base_idx + 1]);
        q[base_idx] = __float2half(q0 * cos_angle - q1 * sin_angle);
        q[base_idx + 1] = __float2half(q0 * sin_angle + q1 * cos_angle);
        
        // Apply rotation to K
        float k0 = __half2float(k[base_idx]);
        float k1 = __half2float(k[base_idx + 1]);
        k[base_idx] = __float2half(k0 * cos_angle - k1 * sin_angle);
        k[base_idx + 1] = __float2half(k0 * sin_angle + k1 * cos_angle);
    }
}

// Feed-forward network with activation functions
__global__ void ffn_activation_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight1,
    const __nv_bfloat16* __restrict__ weight2,
    const half* __restrict__ bias1,
    const half* __restrict__ bias2,
    int batch_size,
    int hidden_dim,
    int ffn_dim,
    bool use_gelu) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * hidden_dim) {
        int batch_idx = idx / hidden_dim;
        int dim_idx = idx % hidden_dim;
        
        // First linear layer
        float sum1 = __half2float(bias1[dim_idx]);
        for (int i = 0; i < hidden_dim; i++) {
            sum1 += __half2float(input[batch_idx * hidden_dim + i]) * 
                   __bfloat162float(weight1[i * ffn_dim + dim_idx]);
        }
        
        // Activation function
        float activated = use_gelu ? gelu(sum1) : silu(sum1);
        
        // Second linear layer
        float sum2 = __half2float(bias2[dim_idx]);
        for (int i = 0; i < ffn_dim; i++) {
            sum2 += activated * __bfloat162float(weight2[i * hidden_dim + dim_idx]);
        }
        
        output[idx] = __float2half(sum2);
    }
}

// LLM Inference kernel implementation
class LLMInferenceKernel : public TypedKernel<half> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    LLMInferenceKernel() {
        cublasCreate(&cublasHandle_);
        cublasSetMathMode(cublasHandle_, CUBLAS_TENSOR_OP_MATH);
    }
    
    ~LLMInferenceKernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // LLM inference parameters
            const int batch_size = 1;  // Typical for inference
            const int seq_len = 2048;  // Common context length
            const int num_heads = 32;
            const int head_dim = 128;
            const int hidden_dim = num_heads * head_dim;
            const int ffn_dim = hidden_dim * 4;
            const int num_layers = 4;  // Reduced for testing
            
            // Allocate device memory for model components
            // Input embeddings
            DeviceBuffer<half> d_input(batch_size * seq_len * hidden_dim);
            
            // QKV projections
            DeviceBuffer<half> d_q(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_k(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_v(batch_size * seq_len * hidden_dim);
            
            // Attention output
            DeviceBuffer<half> d_attn_out(batch_size * seq_len * hidden_dim);
            
            // KV cache
            DeviceBuffer<half> d_cache_k(batch_size * num_layers * seq_len * hidden_dim);
            DeviceBuffer<half> d_cache_v(batch_size * num_layers * seq_len * hidden_dim);
            DeviceBuffer<int> d_cache_pos(batch_size);
            
            // Weights (using mixed precision)
            DeviceBuffer<__nv_bfloat16> d_weight_q(hidden_dim * hidden_dim);
            DeviceBuffer<__nv_bfloat16> d_weight_k(hidden_dim * hidden_dim);
            DeviceBuffer<__nv_bfloat16> d_weight_v(hidden_dim * hidden_dim);
            DeviceBuffer<__nv_bfloat16> d_weight_ffn1(hidden_dim * ffn_dim);
            DeviceBuffer<__nv_bfloat16> d_weight_ffn2(ffn_dim * hidden_dim);
            
            // Biases
            DeviceBuffer<half> d_bias_ffn1(ffn_dim);
            DeviceBuffer<half> d_bias_ffn2(hidden_dim);
            
            // Positional encoding frequencies
            DeviceBuffer<float> d_freqs(head_dim / 2);
            
            // Initialize weights and biases with random values
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
            
            // Initialize input with random embeddings
            curandGenerateUniform(gen, (float*)d_input.get(), 
                                 batch_size * seq_len * hidden_dim / 2);
            
            // Initialize positional encoding frequencies
            float* h_freqs = new float[head_dim / 2];
            for (int i = 0; i < head_dim / 2; i++) {
                h_freqs[i] = 1.0f / powf(10000.0f, 2.0f * i / head_dim);
            }
            cudaMemcpy(d_freqs.get(), h_freqs, sizeof(float) * head_dim / 2, 
                      cudaMemcpyHostToDevice);
            
            // Start timing
            CudaTimer timer;
            timer.start();
            
            // Run inference iterations
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Simulate multi-layer transformer inference
                for (int layer = 0; layer < num_layers; layer++) {
                    // QKV projections with mixed precision
                    dim3 proj_grid((hidden_dim + 15) / 16, (seq_len + 15) / 16);
                    dim3 proj_block(16, 16);
                    mixed_precision_qkv_projection<16><<<proj_grid, proj_block>>>(
                        d_input.get(), d_weight_q.get(), d_weight_k.get(), d_weight_v.get(),
                        d_q.get(), d_k.get(), d_v.get(),
                        seq_len, hidden_dim, hidden_dim);
                    
                    // Apply rotary positional encoding
                    int rope_threads = 256;
                    int rope_blocks = (seq_len * num_heads * head_dim / 2 + rope_threads - 1) / rope_threads;
                    rotary_positional_encoding_kernel<<<rope_blocks, rope_threads>>>(
                        d_q.get(), d_k.get(), d_freqs.get(),
                        seq_len, num_heads, head_dim);
                    
                    // Update KV cache (simulating incremental decoding)
                    int cache_threads = 256;
                    int cache_blocks = (batch_size * num_heads * head_dim + cache_threads - 1) / cache_threads;
                    kv_cache_update_kernel<<<cache_blocks, cache_threads>>>(
                        d_k.get(), d_v.get(),
                        d_cache_k.get() + layer * batch_size * seq_len * hidden_dim,
                        d_cache_v.get() + layer * batch_size * seq_len * hidden_dim,
                        d_cache_pos.get(),
                        batch_size, num_heads, head_dim, seq_len);
                    
                    // Flash attention
                    dim3 attn_grid(seq_len / 64, num_heads);
                    dim3 attn_block(64);
                    size_t shared_mem_size = 3 * 64 * head_dim * sizeof(half) + 64 * 64 * sizeof(float);
                    flash_attention_kernel<64, 128><<<attn_grid, attn_block, shared_mem_size>>>(
                        d_q.get(), d_k.get(), d_v.get(), d_attn_out.get(),
                        seq_len, num_heads, 1.0f / sqrtf(head_dim));
                    
                    // FFN with activation functions
                    int ffn_threads = 256;
                    int ffn_blocks = (batch_size * hidden_dim + ffn_threads - 1) / ffn_threads;
                    bool use_gelu = (layer % 2 == 0);  // Alternate between GELU and SiLU
                    ffn_activation_kernel<<<ffn_blocks, ffn_threads>>>(
                        d_attn_out.get(), d_input.get(),
                        d_weight_ffn1.get(), d_weight_ffn2.get(),
                        d_bias_ffn1.get(), d_bias_ffn2.get(),
                        batch_size, hidden_dim, ffn_dim, use_gelu);
                }
                
                // Memory bandwidth stress - simulate token generation
                cudaMemcpy(d_input.get(), d_attn_out.get(), 
                          sizeof(half) * hidden_dim, cudaMemcpyDeviceToDevice);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate FLOPS for attention and FFN operations
            double attention_flops = (double)num_layers * seq_len * seq_len * hidden_dim * 4.0;
            double ffn_flops = (double)num_layers * seq_len * hidden_dim * ffn_dim * 4.0;
            double total_flops = (attention_flops + ffn_flops) * config.numIterations;
            result.gflops = (total_flops / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth estimation
            double attention_bytes = (double)num_layers * seq_len * hidden_dim * sizeof(half) * 8.0;
            double ffn_bytes = (double)num_layers * seq_len * (hidden_dim + ffn_dim) * sizeof(half) * 2.0;
            double total_bytes = (attention_bytes + ffn_bytes) * config.numIterations;
            result.memoryBandwidthGBps = (total_bytes / 1e9) / (result.executionTimeMs / 1000.0);
            
            result.success = true;
            
            // Set output data for validation
            setOutputData(d_attn_out.get(), batch_size * seq_len * hidden_dim);
            
            // Perform validation if enabled
            if (validationEnabled_ && validationEngine_) {
                performValidation(result, config);
            }
            
            // Perform monitoring if enabled
            if (monitoringEnabled_ && monitoringEngine_) {
                performMonitoring(result, config);
            }
            
            delete[] h_freqs;
            curandDestroyGenerator(gen);
            
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = e.what();
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "LLM Inference Workload Kernel";
    }
    
    std::string getDescription() const override {
        return "Simulates modern LLM inference with attention, KV-cache, mixed precision, and activation functions";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        // Requires FP16 and BF16 support
        return info.supportsFP16() && info.supportsBF16();
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // Rough estimate based on typical LLM inference memory usage
        const int seq_len = 2048;
        const int hidden_dim = 4096;
        const int num_layers = 4;
        const int ffn_dim = hidden_dim * 4;
        
        size_t activation_memory = seq_len * hidden_dim * sizeof(half) * 10;
        size_t weight_memory = hidden_dim * hidden_dim * sizeof(__nv_bfloat16) * 3 +
                              hidden_dim * ffn_dim * sizeof(__nv_bfloat16) * 2;
        size_t cache_memory = num_layers * seq_len * hidden_dim * sizeof(half) * 2;
        
        return activation_memory + weight_memory + cache_memory;
    }
};

// Register kernel
REGISTER_KERNEL(LLMInferenceKernel)