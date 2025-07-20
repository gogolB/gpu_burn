#include "kernel_interface.h"
#include "gpu_utils.h"
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

// Layer normalization kernel for training
__global__ void layer_norm_forward_kernel(
    const half* __restrict__ input,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    half* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int batch_size,
    int hidden_dim,
    float eps) {
    
    extern __shared__ float shared[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // Compute mean
    float thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        thread_sum += __half2float(input[batch_idx * hidden_dim + i]);
    }
    
    // Reduce sum within block
    shared[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float batch_mean = shared[0] / hidden_dim;
    if (tid == 0) {
        mean[batch_idx] = batch_mean;
    }
    __syncthreads();
    
    // Compute variance
    thread_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float diff = __half2float(input[batch_idx * hidden_dim + i]) - batch_mean;
        thread_sum += diff * diff;
    }
    
    shared[tid] = thread_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float batch_var = shared[0] / hidden_dim;
    float batch_rstd = rsqrtf(batch_var + eps);
    if (tid == 0) {
        rstd[batch_idx] = batch_rstd;
    }
    __syncthreads();
    
    // Normalize and apply affine transformation
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = (__half2float(input[batch_idx * hidden_dim + i]) - batch_mean) * batch_rstd;
        float scaled = normalized * __half2float(gamma[i]) + __half2float(beta[i]);
        output[batch_idx * hidden_dim + i] = __float2half(scaled);
    }
}

// Layer normalization backward kernel
__global__ void layer_norm_backward_kernel(
    const half* __restrict__ grad_output,
    const half* __restrict__ input,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const half* __restrict__ gamma,
    half* __restrict__ grad_input,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int batch_size,
    int hidden_dim) {
    
    extern __shared__ float shared[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    float batch_mean = mean[batch_idx];
    float batch_rstd = rstd[batch_idx];
    
    // Compute gradients for gamma and beta
    float thread_grad_gamma = 0.0f;
    float thread_grad_beta = 0.0f;
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float normalized = (__half2float(input[batch_idx * hidden_dim + i]) - batch_mean) * batch_rstd;
        float grad_out = __half2float(grad_output[batch_idx * hidden_dim + i]);
        
        thread_grad_gamma += grad_out * normalized;
        thread_grad_beta += grad_out;
    }
    
    // Atomic add to global memory for gradient accumulation
    if (thread_grad_gamma != 0.0f) {
        atomicAdd(&grad_gamma[tid], thread_grad_gamma);
    }
    if (thread_grad_beta != 0.0f) {
        atomicAdd(&grad_beta[tid], thread_grad_beta);
    }
    
    // Compute gradient for input
    float grad_mean = 0.0f;
    float grad_var = 0.0f;
    
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float grad_out = __half2float(grad_output[batch_idx * hidden_dim + i]);
        float gamma_val = __half2float(gamma[i]);
        grad_mean += grad_out * gamma_val;
        float normalized = (__half2float(input[batch_idx * hidden_dim + i]) - batch_mean) * batch_rstd;
        grad_var += grad_out * gamma_val * normalized;
    }
    
    shared[tid] = grad_mean;
    shared[tid + blockDim.x] = grad_var;
    __syncthreads();
    
    // Reduce gradients
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
            shared[tid + blockDim.x] += shared[tid + blockDim.x + stride];
        }
        __syncthreads();
    }
    
    grad_mean = shared[0] / hidden_dim;
    grad_var = shared[blockDim.x] * batch_rstd * batch_rstd * batch_rstd / hidden_dim;
    
    // Apply chain rule
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float grad_out = __half2float(grad_output[batch_idx * hidden_dim + i]);
        float gamma_val = __half2float(gamma[i]);
        float input_val = __half2float(input[batch_idx * hidden_dim + i]);
        
        float grad = gamma_val * batch_rstd * (grad_out - grad_mean - (input_val - batch_mean) * grad_var);
        grad_input[batch_idx * hidden_dim + i] = __float2half(grad);
    }
}

// Attention backward kernel with gradient accumulation
template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void attention_backward_kernel(
    const half* __restrict__ grad_output,
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ attn_weights,
    half* __restrict__ grad_q,
    half* __restrict__ grad_k,
    half* __restrict__ grad_v,
    int seq_len,
    int num_heads,
    float scale) {
    
    extern __shared__ float shared_mem[];
    
    int head_idx = blockIdx.y;
    int pos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;
    
    if (pos >= seq_len) return;
    
    // Gradient accumulation buffers
    float grad_q_acc[HEAD_DIM] = {0.0f};
    float grad_k_acc[HEAD_DIM] = {0.0f};
    float grad_v_acc[HEAD_DIM] = {0.0f};
    
    // Backward through attention
    for (int kv_block = 0; kv_block < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; kv_block++) {
        int kv_pos = kv_block * BLOCK_SIZE + tid;
        
        if (kv_pos < seq_len) {
            // Load attention weight
            float attn_weight = __half2float(attn_weights[(head_idx * seq_len + pos) * seq_len + kv_pos]);
            
            // Gradient w.r.t V
            for (int d = 0; d < HEAD_DIM; d++) {
                float grad_out = __half2float(grad_output[(head_idx * seq_len + pos) * HEAD_DIM + d]);
                grad_v_acc[d] += attn_weight * grad_out;
            }
            
            // Gradient w.r.t attention weights
            float grad_attn = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                grad_attn += __half2float(grad_output[(head_idx * seq_len + pos) * HEAD_DIM + d]) *
                            __half2float(v[(head_idx * seq_len + kv_pos) * HEAD_DIM + d]);
            }
            
            // Gradient w.r.t Q and K
            grad_attn *= scale * attn_weight;
            for (int d = 0; d < HEAD_DIM; d++) {
                grad_q_acc[d] += grad_attn * __half2float(k[(head_idx * seq_len + kv_pos) * HEAD_DIM + d]);
                grad_k_acc[d] += grad_attn * __half2float(q[(head_idx * seq_len + pos) * HEAD_DIM + d]);
            }
        }
    }
    
    // Write gradients
    if (pos < seq_len) {
        for (int d = 0; d < HEAD_DIM; d++) {
            int idx = (head_idx * seq_len + pos) * HEAD_DIM + d;
            grad_q[idx] = __float2half(grad_q_acc[d]);
            grad_k[idx] = __float2half(grad_k_acc[d]);
            grad_v[idx] = __float2half(grad_v_acc[d]);
        }
    }
}

// Adam optimizer kernel with mixed precision
__global__ void adam_optimizer_kernel(
    const half* __restrict__ gradients,
    float* __restrict__ params,           // FP32 master weights
    __nv_bfloat16* __restrict__ params_bf16,  // BF16 copy for forward pass
    float* __restrict__ m,                // First moment
    float* __restrict__ v,                // Second moment
    float* __restrict__ m_hat,
    float* __restrict__ v_hat,
    int num_params,
    float lr,
    float beta1,
    float beta2,
    float eps,
    int step,
    float weight_decay,
    bool use_grad_scaling,
    float grad_scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_params) {
        // Load gradient and apply gradient scaling if needed
        float grad = __half2float(gradients[idx]);
        if (use_grad_scaling) {
            grad /= grad_scale;
        }
        
        // Apply weight decay
        float param_val = params[idx];
        grad += weight_decay * param_val;
        
        // Update biased first moment estimate
        float m_val = m[idx];
        m_val = beta1 * m_val + (1.0f - beta1) * grad;
        m[idx] = m_val;
        
        // Update biased second moment estimate
        float v_val = v[idx];
        v_val = beta2 * v_val + (1.0f - beta2) * grad * grad;
        v[idx] = v_val;
        
        // Compute bias-corrected moment estimates
        float m_hat_val = m_val / (1.0f - powf(beta1, step));
        float v_hat_val = v_val / (1.0f - powf(beta2, step));
        
        // Store for potential analysis
        if (m_hat != nullptr) m_hat[idx] = m_hat_val;
        if (v_hat != nullptr) v_hat[idx] = v_hat_val;
        
        // Update parameters
        param_val -= lr * m_hat_val / (sqrtf(v_hat_val) + eps);
        
        // Gradient clipping
        const float max_grad_norm = 1.0f;
        float grad_norm = fabsf(m_hat_val / (sqrtf(v_hat_val) + eps));
        if (grad_norm > max_grad_norm) {
            param_val = params[idx] - lr * max_grad_norm * (m_hat_val / grad_norm);
        }
        
        // Store FP32 master weight
        params[idx] = param_val;
        
        // Convert to BF16 for forward pass
        params_bf16[idx] = __float2bfloat16(param_val);
    }
}

// Gradient accumulation and all-reduce simulation kernel
__global__ void gradient_accumulation_kernel(
    const half* __restrict__ local_gradients,
    float* __restrict__ accumulated_gradients,
    int num_params,
    int accumulation_steps,
    bool do_allreduce) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_params) {
        // Accumulate gradients
        float acc_grad = accumulated_gradients[idx];
        acc_grad += __half2float(local_gradients[idx]);
        
        if (do_allreduce) {
            // Simulate all-reduce by adding noise (in real scenario, this would be MPI/NCCL)
            acc_grad += 0.001f * sinf((float)idx);
            
            // Average over accumulation steps
            acc_grad /= accumulation_steps;
        }
        
        accumulated_gradients[idx] = acc_grad;
    }
}

// Activation checkpointing recomputation kernel
__global__ void activation_checkpoint_recompute_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const __nv_bfloat16* __restrict__ weight,
    int batch_size,
    int input_dim,
    int output_dim,
    bool save_for_backward) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / output_dim;
    int out_idx = idx % output_dim;
    
    if (batch_idx < batch_size && out_idx < output_dim) {
        float sum = 0.0f;
        
        // Recompute forward pass
        for (int i = 0; i < input_dim; i++) {
            sum += __half2float(input[batch_idx * input_dim + i]) * 
                   __bfloat162float(weight[i * output_dim + out_idx]);
        }
        
        // Apply activation (ReLU for simplicity)
        sum = fmaxf(0.0f, sum);
        
        output[idx] = __float2half(sum);
        
        // Simulate memory pattern of saving activations for backward
        if (save_for_backward && (idx % 4 == 0)) {
            // Force memory transaction
            atomicAdd((float*)&output[0], 0.0f);
        }
    }
}

// Mixed precision loss scaling kernel
__global__ void loss_scaling_kernel(
    half* __restrict__ loss,
    half* __restrict__ gradients,
    float* __restrict__ scale_factor,
    int num_elements,
    bool check_overflow) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_elements) {
        float grad = __half2float(gradients[idx]);
        float scale = *scale_factor;
        
        // Apply loss scaling
        grad *= scale;
        
        // Check for overflow/underflow
        if (check_overflow) {
            if (!isfinite(grad) || fabsf(grad) > 65504.0f) {
                // Overflow detected - reduce scale
                atomicMin((int*)scale_factor, __float_as_int(scale * 0.5f));
                grad = 0.0f;  // Zero out gradient
            } else if (fabsf(grad) < 1e-7f && fabsf(grad) > 0.0f) {
                // Underflow detected - increase scale
                atomicMax((int*)scale_factor, __float_as_int(scale * 2.0f));
            }
        }
        
        gradients[idx] = __float2half(grad);
    }
}

// LLM Training kernel implementation
class LLMTrainingKernel : public TypedKernel<half> {
private:
    cublasHandle_t cublasHandle_;
    
public:
    LLMTrainingKernel() {
        cublasCreate(&cublasHandle_);
        cublasSetMathMode(cublasHandle_, CUBLAS_TENSOR_OP_MATH);
    }
    
    ~LLMTrainingKernel() {
        cublasDestroy(cublasHandle_);
    }
    
    KernelResult execute(const KernelConfig& config) override {
        KernelResult result;
        
        try {
            CUDA_CHECK(cudaSetDevice(config.deviceId));
            
            // LLM training parameters
            const int batch_size = 8;      // Mini-batch size
            const int seq_len = 512;       // Sequence length for training
            const int num_heads = 32;
            const int head_dim = 128;
            const int hidden_dim = num_heads * head_dim;
            const int ffn_dim = hidden_dim * 4;
            const int num_layers = 2;      // Reduced for testing
            const int vocab_size = 50000;
            const int accumulation_steps = 4;
            
            // Allocate device memory for training components
            // Forward pass tensors
            DeviceBuffer<half> d_input(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_output(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_q(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_k(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_v(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_attn_out(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_attn_weights(batch_size * num_heads * seq_len * seq_len);
            
            // Layer norm parameters
            DeviceBuffer<half> d_ln_gamma(hidden_dim);
            DeviceBuffer<half> d_ln_beta(hidden_dim);
            DeviceBuffer<float> d_ln_mean(batch_size);
            DeviceBuffer<float> d_ln_rstd(batch_size);
            
            // Backward pass tensors
            DeviceBuffer<half> d_grad_output(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_grad_input(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_grad_q(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_grad_k(batch_size * seq_len * hidden_dim);
            DeviceBuffer<half> d_grad_v(batch_size * seq_len * hidden_dim);
            DeviceBuffer<float> d_grad_ln_gamma(hidden_dim);
            DeviceBuffer<float> d_grad_ln_beta(hidden_dim);
            
            // Model weights (mixed precision)
            DeviceBuffer<float> d_params_fp32(hidden_dim * hidden_dim * 3);  // Q, K, V projections
            DeviceBuffer<__nv_bfloat16> d_params_bf16(hidden_dim * hidden_dim * 3);
            
            // Optimizer states
            DeviceBuffer<float> d_adam_m(hidden_dim * hidden_dim * 3);
            DeviceBuffer<float> d_adam_v(hidden_dim * hidden_dim * 3);
            DeviceBuffer<float> d_adam_m_hat(hidden_dim * hidden_dim * 3);
            DeviceBuffer<float> d_adam_v_hat(hidden_dim * hidden_dim * 3);
            
            // Gradient accumulation buffer
            DeviceBuffer<float> d_accumulated_grads(hidden_dim * hidden_dim * 3);
            
            // Loss scaling
            DeviceBuffer<float> d_loss_scale(1);
            float initial_loss_scale = 1024.0f;
            cudaMemcpy(d_loss_scale.get(), &initial_loss_scale, sizeof(float), cudaMemcpyHostToDevice);
            
            // Initialize parameters
            curandGenerator_t gen;
            curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
            
            // Initialize with Xavier/He initialization pattern
            float* h_params = new float[hidden_dim * hidden_dim * 3];
            float init_scale = sqrtf(2.0f / hidden_dim);
            for (int i = 0; i < hidden_dim * hidden_dim * 3; i++) {
                h_params[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 2.0f * init_scale;
            }
            cudaMemcpy(d_params_fp32.get(), h_params, sizeof(float) * hidden_dim * hidden_dim * 3, 
                      cudaMemcpyHostToDevice);
            
            // Initialize layer norm parameters
            float* h_gamma = new float[hidden_dim];
            float* h_beta = new float[hidden_dim];
            for (int i = 0; i < hidden_dim; i++) {
                h_gamma[i] = 1.0f;
                h_beta[i] = 0.0f;
            }
            cudaMemcpy(d_ln_gamma.get(), h_gamma, sizeof(half) * hidden_dim, cudaMemcpyHostToDevice);
            cudaMemcpy(d_ln_beta.get(), h_beta, sizeof(half) * hidden_dim, cudaMemcpyHostToDevice);
            
            // Adam optimizer parameters
            float lr = 0.0001f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            float weight_decay = 0.01f;
            
            // Start timing
            CudaTimer timer;
            timer.start();
            
            // Training iterations
            for (size_t iter = 0; iter < config.numIterations; iter++) {
                // Simulate gradient accumulation
                for (int accum_step = 0; accum_step < accumulation_steps; accum_step++) {
                    // Forward pass
                    // Layer normalization
                    dim3 ln_grid(batch_size);
                    dim3 ln_block(256);
                    size_t ln_shared = ln_block.x * sizeof(float) * 2;
                    layer_norm_forward_kernel<<<ln_grid, ln_block, ln_shared>>>(
                        d_input.get(), d_ln_gamma.get(), d_ln_beta.get(), d_output.get(),
                        d_ln_mean.get(), d_ln_rstd.get(), batch_size, hidden_dim, 1e-5f);
                    
                    // Multi-head attention (using cuBLAS for efficiency)
                    const half alpha = __float2half(1.0f);
                    const half beta = __float2half(0.0f);
                    
                    // Q, K, V projections
                    cublasHgemm(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
                               hidden_dim, batch_size * seq_len, hidden_dim,
                               &alpha,
                               (const half*)d_params_bf16.get(), hidden_dim,
                               d_output.get(), hidden_dim,
                               &beta,
                               d_q.get(), hidden_dim);
                    
                    // Backward pass
                    // Initialize gradient from loss
                    curandGenerateUniform(gen, (float*)d_grad_output.get(), 
                                        batch_size * seq_len * hidden_dim / 2);
                    
                    // Layer norm backward
                    layer_norm_backward_kernel<<<ln_grid, ln_block, ln_shared>>>(
                        d_grad_output.get(), d_input.get(), d_ln_mean.get(), d_ln_rstd.get(),
                        d_ln_gamma.get(), d_grad_input.get(), d_grad_ln_gamma.get(),
                        d_grad_ln_beta.get(), batch_size, hidden_dim);
                    
                    // Attention backward
                    dim3 attn_back_grid(seq_len / 64, num_heads);
                    dim3 attn_back_block(64);
                    attention_backward_kernel<64, 128><<<attn_back_grid, attn_back_block>>>(
                        d_grad_output.get(), d_q.get(), d_k.get(), d_v.get(),
                        d_attn_weights.get(), d_grad_q.get(), d_grad_k.get(), d_grad_v.get(),
                        seq_len, num_heads, 1.0f / sqrtf(head_dim));
                    
                    // Gradient accumulation
                    int grad_accum_threads = 256;
                    int grad_accum_blocks = (hidden_dim * hidden_dim * 3 + grad_accum_threads - 1) / grad_accum_threads;
                    bool do_allreduce = (accum_step == accumulation_steps - 1);
                    gradient_accumulation_kernel<<<grad_accum_blocks, grad_accum_threads>>>(
                        d_grad_input.get(), d_accumulated_grads.get(),
                        hidden_dim * hidden_dim * 3, accumulation_steps, do_allreduce);
                    
                    // Activation checkpointing simulation
                    if (iter % 2 == 0) {  // Simulate checkpointing every other layer
                        int ckpt_threads = 256;
                        int ckpt_blocks = (batch_size * hidden_dim + ckpt_threads - 1) / ckpt_threads;
                        activation_checkpoint_recompute_kernel<<<ckpt_blocks, ckpt_threads>>>(
                            d_input.get(), d_output.get(), d_params_bf16.get(),
                            batch_size, hidden_dim, hidden_dim, true);
                    }
                }
                
                // Loss scaling
                int scale_threads = 256;
                int scale_blocks = (hidden_dim * hidden_dim * 3 + scale_threads - 1) / scale_threads;
                loss_scaling_kernel<<<scale_blocks, scale_threads>>>(
                    d_grad_output.get(), d_grad_input.get(), d_loss_scale.get(),
                    hidden_dim * hidden_dim * 3, true);
                
                // Adam optimizer update
                int adam_threads = 256;
                int adam_blocks = (hidden_dim * hidden_dim * 3 + adam_threads - 1) / adam_threads;
                adam_optimizer_kernel<<<adam_blocks, adam_threads>>>(
                    d_grad_input.get(), d_params_fp32.get(), d_params_bf16.get(),
                    d_adam_m.get(), d_adam_v.get(), d_adam_m_hat.get(), d_adam_v_hat.get(),
                    hidden_dim * hidden_dim * 3,
                    lr, beta1, beta2, eps, iter + 1, weight_decay,
                    true, initial_loss_scale);
                
                // Clear gradient accumulation buffer
                cudaMemset(d_accumulated_grads.get(), 0, sizeof(float) * hidden_dim * hidden_dim * 3);
            }
            
            timer.stop();
            CUDA_CHECK_KERNEL();
            
            // Calculate performance metrics
            result.executionTimeMs = timer.getElapsedMs();
            
            // Estimate FLOPS for training (forward + backward + optimizer)
            double forward_flops = (double)num_layers * batch_size * seq_len * hidden_dim * hidden_dim * 6.0;
            double backward_flops = forward_flops * 2.0;  // Roughly 2x forward
            double optimizer_flops = (double)hidden_dim * hidden_dim * 3 * 10.0;  // Adam operations
            double total_flops = (forward_flops + backward_flops + optimizer_flops) * 
                                config.numIterations * accumulation_steps;
            result.gflops = (total_flops / 1e9) / (result.executionTimeMs / 1000.0);
            
            // Memory bandwidth estimation
            double forward_bytes = (double)batch_size * seq_len * hidden_dim * sizeof(half) * 20.0;
            double backward_bytes = forward_bytes * 2.5;  // More memory intensive
            double optimizer_bytes = (double)hidden_dim * hidden_dim * 3 * sizeof(float) * 6.0;  // Read/write multiple buffers
            double total_bytes = (forward_bytes + backward_bytes + optimizer_bytes) * 
                               config.numIterations * accumulation_steps;
            result.memoryBandwidthGBps = (total_bytes / 1e9) / (result.executionTimeMs / 1000.0);
            
            result.success = true;
            
            // Set output data for validation
            setOutputData(d_output.get(), batch_size * seq_len * hidden_dim);
            
            // Perform validation if enabled
            if (validationEnabled_ && validationEngine_) {
                performValidation(result, config);
            }
            
            // Perform monitoring if enabled
            if (monitoringEnabled_ && monitoringEngine_) {
                performMonitoring(result, config);
            }
            
            delete[] h_params;
            delete[] h_gamma;
            delete[] h_beta;
            curandDestroyGenerator(gen);
            
        } catch (const std::exception& e) {
            result.success = false;
            result.errorMessage = e.what();
        }
        
        return result;
    }
    
    std::string getName() const override {
        return "LLM Training Workload Kernel";
    }
    
    std::string getDescription() const override {
        return "Simulates full LLM training with forward/backward passes, mixed precision, Adam optimizer, and gradient accumulation";
    }
    
    bool isSupported(int deviceId) const override {
        GpuInfo info = getGpuInfo(deviceId);
        // Requires FP16, BF16, and sufficient memory
        return info.supportsFP16() && info.supportsBF16() && 
               (info.totalMemory >= 8ULL * 1024 * 1024 * 1024);  // At least 8GB
    }
    
    size_t getMemoryRequirement(const KernelConfig& config) const override {
        // Rough estimate based on typical LLM training memory usage
        const int batch_size = 8;
        const int seq_len = 512;
        const int hidden_dim = 4096;
        const int num_layers = 2;
        
        // Model parameters + optimizer states + activations + gradients
        size_t param_memory = hidden_dim * hidden_dim * 3 * sizeof(float) * 3;  // Params + Adam states
        size_t activation_memory = batch_size * seq_len * hidden_dim * sizeof(half) * 10;
        size_t gradient_memory = activation_memory;
        
        return param_memory + activation_memory + gradient_memory;
    }
};

// Register kernel
REGISTER_KERNEL(LLMTrainingKernel)