#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                     std::to_string(__LINE__) + " - " + \
                                     cudaGetErrorString(error)); \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel launch error: ") + \
                                     cudaGetErrorString(error)); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA kernel execution error: ") + \
                                     cudaGetErrorString(error)); \
        } \
    } while(0)

// GPU device information
struct GpuInfo {
    std::string name;
    int computeCapabilityMajor;
    int computeCapabilityMinor;
    size_t totalMemory;
    size_t sharedMemPerBlock;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    int maxThreadsPerMultiProcessor;
    bool tensorCoreSupport;
    
    bool supportsArch(int major, int minor) const {
        return (computeCapabilityMajor > major) || 
               (computeCapabilityMajor == major && computeCapabilityMinor >= minor);
    }
    
    bool supportsFP16() const {
        return supportsArch(5, 3); // Kepler or newer
    }
    
    bool supportsBF16() const {
        return supportsArch(8, 0); // Ampere or newer
    }
    
    bool supportsFP8() const {
        return supportsArch(8, 9); // Ada Lovelace or newer
    }
    
    bool supportsTensorCores() const {
        return tensorCoreSupport && supportsArch(7, 0); // Volta or newer
    }
};

// Get GPU information
inline GpuInfo getGpuInfo(int deviceId = 0) {
    GpuInfo info;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    
    info.name = prop.name;
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.totalMemory = prop.totalGlobalMem;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    
    // Check for tensor core support based on compute capability
    info.tensorCoreSupport = (prop.major >= 7); // Volta and newer
    
    return info;
}

// Architecture detection helpers
enum class GpuArchitecture {
    UNKNOWN,
    VOLTA,      // SM 7.0, 7.2
    TURING,     // SM 7.5
    AMPERE,     // SM 8.0, 8.6
    ADA,        // SM 8.9
    HOPPER,     // SM 9.0
    BLACKWELL   // SM 10.0 (future)
};

inline GpuArchitecture detectArchitecture(const GpuInfo& info) {
    int sm = info.computeCapabilityMajor * 10 + info.computeCapabilityMinor;
    
    if (sm >= 100) return GpuArchitecture::BLACKWELL;
    if (sm >= 90) return GpuArchitecture::HOPPER;
    if (sm >= 89) return GpuArchitecture::ADA;
    if (sm >= 80) return GpuArchitecture::AMPERE;
    if (sm >= 75) return GpuArchitecture::TURING;
    if (sm >= 70) return GpuArchitecture::VOLTA;
    
    return GpuArchitecture::UNKNOWN;
}

inline std::string architectureToString(GpuArchitecture arch) {
    switch (arch) {
        case GpuArchitecture::VOLTA: return "Volta";
        case GpuArchitecture::TURING: return "Turing";
        case GpuArchitecture::AMPERE: return "Ampere";
        case GpuArchitecture::ADA: return "Ada Lovelace";
        case GpuArchitecture::HOPPER: return "Hopper";
        case GpuArchitecture::BLACKWELL: return "Blackwell";
        default: return "Unknown";
    }
}

// Performance timing utilities
class CudaTimer {
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
    }
    
    float getElapsedMs() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Memory allocation helpers
template<typename T>
class DeviceBuffer {
private:
    T* ptr_;
    size_t size_;
    
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    
    explicit DeviceBuffer(size_t size) : ptr_(nullptr), size_(size) {
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size * sizeof(T)));
        }
    }
    
    ~DeviceBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }
    
    // Disable copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Enable move
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return size_; }
    
    void resize(size_t newSize) {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = newSize;
        if (size_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
        }
    }
};

#endif // GPU_UTILS_H