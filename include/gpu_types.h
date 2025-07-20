#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// Standard floating point types
using fp32_t = float;
using fp64_t = double;

// Half precision types
using fp16_t = __half;
using bf16_t = __nv_bfloat16;

// 8-bit floating point types (for Ada/Hopper/Blackwell)
// Note: These are experimental types, actual support depends on GPU architecture
struct fp8_e4m3_t {
    uint8_t data;
    
    __device__ __host__ fp8_e4m3_t() : data(0) {}
    __device__ __host__ fp8_e4m3_t(float val) {
        // Simplified conversion - actual implementation would be more complex
        data = static_cast<uint8_t>(val * 16.0f);
    }
    __device__ __host__ operator float() const {
        return static_cast<float>(data) / 16.0f;
    }
};

struct fp8_e5m2_t {
    uint8_t data;
    
    __device__ __host__ fp8_e5m2_t() : data(0) {}
    __device__ __host__ fp8_e5m2_t(float val) {
        // Simplified conversion - actual implementation would be more complex
        data = static_cast<uint8_t>(val * 4.0f);
    }
    __device__ __host__ operator float() const {
        return static_cast<float>(data) / 4.0f;
    }
};

using fp8_t = fp8_e4m3_t;  // Default FP8 type
using bf8_t = fp8_e5m2_t;  // BFloat8 variant

// 4-bit floating point (experimental)
struct fp4_t {
    uint8_t data : 4;
    
    __device__ __host__ fp4_t() : data(0) {}
    __device__ __host__ fp4_t(float val) {
        // Very simplified conversion
        data = static_cast<uint8_t>(val * 2.0f) & 0xF;
    }
    __device__ __host__ operator float() const {
        return static_cast<float>(data) / 2.0f;
    }
};

// Integer types
using uint8_t = unsigned char;
using uint16_t = unsigned short;
using int8_t = signed char;
using int16_t = short;

// Precision type enumeration
enum class PrecisionType {
    FP64,
    FP32,
    FP16,
    BF16,
    FP8,
    BF8,
    FP4,
    UINT16,
    UINT8,
    INT16,
    INT8
};

// Helper functions for type information
template<typename T>
struct TypeInfo {
    static constexpr const char* name = "unknown";
    static constexpr PrecisionType precision = PrecisionType::FP32;
};

template<> struct TypeInfo<fp64_t> {
    static constexpr const char* name = "FP64";
    static constexpr PrecisionType precision = PrecisionType::FP64;
};

template<> struct TypeInfo<fp32_t> {
    static constexpr const char* name = "FP32";
    static constexpr PrecisionType precision = PrecisionType::FP32;
};

template<> struct TypeInfo<fp16_t> {
    static constexpr const char* name = "FP16";
    static constexpr PrecisionType precision = PrecisionType::FP16;
};

template<> struct TypeInfo<bf16_t> {
    static constexpr const char* name = "BF16";
    static constexpr PrecisionType precision = PrecisionType::BF16;
};

template<> struct TypeInfo<fp8_t> {
    static constexpr const char* name = "FP8";
    static constexpr PrecisionType precision = PrecisionType::FP8;
};

template<> struct TypeInfo<bf8_t> {
    static constexpr const char* name = "BF8";
    static constexpr PrecisionType precision = PrecisionType::BF8;
};

template<> struct TypeInfo<fp4_t> {
    static constexpr const char* name = "FP4";
    static constexpr PrecisionType precision = PrecisionType::FP4;
};

template<> struct TypeInfo<uint16_t> {
    static constexpr const char* name = "UINT16";
    static constexpr PrecisionType precision = PrecisionType::UINT16;
};

template<> struct TypeInfo<uint8_t> {
    static constexpr const char* name = "UINT8";
    static constexpr PrecisionType precision = PrecisionType::UINT8;
};

#endif // GPU_TYPES_H