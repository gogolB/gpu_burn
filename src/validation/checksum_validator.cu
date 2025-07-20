#include "checksum_validator.h"
#include "validation_type_converter.h"
#include <cuda_runtime.h>
#include <cstring>
#include <sstream>

// CRC32 polynomial
#define CRC32_POLYNOMIAL 0xEDB88320
#define CRC64_POLYNOMIAL 0xC96C5795D7870F42ULL

// Define the constant memory tables (actual definitions, not declarations)
__constant__ uint32_t d_crc32Table[256];
__constant__ uint64_t d_crc64Table[256];

// Initialize CRC tables on host
static uint32_t h_crc32Table[256];
static uint64_t h_crc64Table[256];
static bool tablesInitialized = false;

void initializeCRCTables() {
    if (tablesInitialized) return;
    
    // Initialize CRC32 table
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? CRC32_POLYNOMIAL : 0);
        }
        h_crc32Table[i] = crc;
    }
    
    // Initialize CRC64 table
    for (uint64_t i = 0; i < 256; i++) {
        uint64_t crc = i;
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ ((crc & 1) ? CRC64_POLYNOMIAL : 0);
        }
        h_crc64Table[i] = crc;
    }
    
    // Copy tables to device constant memory
    cudaMemcpyToSymbol(d_crc32Table, h_crc32Table, sizeof(h_crc32Table));
    cudaMemcpyToSymbol(d_crc64Table, h_crc64Table, sizeof(h_crc64Table));
    
    tablesInitialized = true;
}

// CRC32 kernel
__global__ void crc32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result) {
    extern __shared__ uint32_t sharedCRC[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    uint32_t crc = 0xFFFFFFFF;
    
    // Process bytes with stride
    for (size_t i = gid; i < numBytes; i += stride) {
        uint8_t byte = data[i];
        crc = (crc >> 8) ^ d_crc32Table[(crc ^ byte) & 0xFF];
    }
    
    sharedCRC[tid] = crc;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Combine CRCs (simplified - real CRC combination is more complex)
            sharedCRC[tid] ^= sharedCRC[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicXor(result, ~sharedCRC[0]);
    }
}

// Fletcher32 kernel
__global__ void fletcher32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result) {
    extern __shared__ uint32_t sharedSums[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    uint32_t sum1 = 0;
    uint32_t sum2 = 0;
    
    // Process data
    for (size_t i = gid; i < numBytes; i += stride) {
        sum1 = (sum1 + data[i]) % 65535;
        sum2 = (sum2 + sum1) % 65535;
    }
    
    sharedSums[tid * 2] = sum1;
    sharedSums[tid * 2 + 1] = sum2;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSums[tid * 2] = (sharedSums[tid * 2] + sharedSums[(tid + s) * 2]) % 65535;
            sharedSums[tid * 2 + 1] = (sharedSums[tid * 2 + 1] + sharedSums[(tid + s) * 2 + 1]) % 65535;
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        uint32_t fletcher = (sharedSums[1] << 16) | sharedSums[0];
        atomicAdd(result, fletcher);
    }
}

// Adler32 kernel
__global__ void adler32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result) {
    extern __shared__ uint32_t sharedAdler[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    uint32_t a = 1;
    uint32_t b = 0;
    const uint32_t MOD_ADLER = 65521;
    
    for (size_t i = gid; i < numBytes; i += stride) {
        a = (a + data[i]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    sharedAdler[tid * 2] = a;
    sharedAdler[tid * 2 + 1] = b;
    __syncthreads();
    
    // Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedAdler[tid * 2] = (sharedAdler[tid * 2] + sharedAdler[(tid + s) * 2]) % MOD_ADLER;
            sharedAdler[tid * 2 + 1] = (sharedAdler[tid * 2 + 1] + sharedAdler[(tid + s) * 2 + 1]) % MOD_ADLER;
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        uint32_t adler = (sharedAdler[1] << 16) | sharedAdler[0];
        atomicAdd(result, adler);
    }
}

ChecksumValidator::ChecksumValidator(ChecksumType type)
    : checksumType_(type)
    , expectedChecksum_(0)
    , hasExpected_(false)
    , continuousMode_(false)
    , d_checksum_(nullptr)
    , d_scratchBuffer_(nullptr)
    , scratchBufferSize_(0)
    , maxHistorySize_(100) {
    
    initializeCRCTables();
}

ChecksumValidator::~ChecksumValidator() {
    cleanup();
}

void ChecksumValidator::setup(const KernelConfig& config) {
    // Allocate device memory for checksum result
    if (d_checksum_ == nullptr) {
        cudaMalloc(&d_checksum_, sizeof(uint64_t));
    }
    
    // Allocate scratch buffer for intermediate computations
    size_t requiredSize = config.matrixSize * config.matrixSize * sizeof(float);
    if (scratchBufferSize_ < requiredSize) {
        if (d_scratchBuffer_) {
            cudaFree(d_scratchBuffer_);
        }
        cudaMalloc(&d_scratchBuffer_, requiredSize);
        scratchBufferSize_ = requiredSize;
    }
}

void ChecksumValidator::cleanup() {
    if (d_checksum_) {
        cudaFree(d_checksum_);
        d_checksum_ = nullptr;
    }
    if (d_scratchBuffer_) {
        cudaFree(d_scratchBuffer_);
        d_scratchBuffer_ = nullptr;
        scratchBufferSize_ = 0;
    }
    checksumHistory_.clear();
}

ValidationResult ChecksumValidator::validate(
    const void* data,
    size_t numElements,
    size_t elementSize,
    const KernelConfig& config) {
    
    ValidationResult result;
    result.method = ValidationType::CHECKSUM;
    
    // For non-CPU types, we have two options:
    // 1. Convert to float32 first for consistent checksums
    // 2. Compute checksum on raw bytes (may vary between runs due to precision)
    
    std::unique_ptr<float[]> convertedData;
    const void* dataToChecksum = data;
    size_t numBytes = numElements * elementSize;
    
    // Option 1: Convert non-CPU types for consistent checksums
    if (ValidationTypeConverter::requiresConversion(elementSize)) {
        try {
            convertedData = ValidationTypeConverter::convertToFloat32(data, numElements, elementSize);
            dataToChecksum = convertedData.get();
            numBytes = numElements * sizeof(float);
            
            // Note the conversion in results
            result.errorDetails = "Checksum computed on float32-converted data";
        } catch (const std::exception& e) {
            // If conversion fails, fall back to raw byte checksum
            dataToChecksum = data;
            result.errorDetails = "Using raw byte checksum for non-CPU type";
        }
    }
    
    uint64_t computedChecksum = computeChecksum(dataToChecksum, numBytes);
    
    if (continuousMode_) {
        updateHistory(computedChecksum);
        
        // In continuous mode, check for anomalies in checksum history
        if (detectAnomalyInHistory()) {
            result.passed = false;
            result.errorDetails = "Checksum anomaly detected in continuous monitoring";
            result.confidence = 0.5;  // Medium confidence for anomaly detection
        }
    }
    
    if (hasExpected_) {
        result.passed = validateChecksum(computedChecksum, expectedChecksum_);
        if (!result.passed) {
            std::stringstream ss;
            ss << "Checksum mismatch: expected 0x" << std::hex << expectedChecksum_ 
               << ", got 0x" << computedChecksum;
            result.errorDetails = ss.str();
            result.confidence = 1.0;  // High confidence for direct mismatch
            result.corruptedElements = 1;  // At least one element corrupted
        }
    } else if (!continuousMode_) {
        // Store checksum for future reference
        expectedChecksum_ = computedChecksum;
        hasExpected_ = true;
        result.passed = true;
        result.confidence = 1.0;
    }
    
    return result;
}

uint64_t ChecksumValidator::computeChecksum(const void* data, size_t numBytes) {
    switch (checksumType_) {
        case ChecksumType::CRC32:
            return computeCRC32(data, numBytes);
        case ChecksumType::CRC64:
            return computeCRC64(data, numBytes);
        case ChecksumType::FLETCHER32:
            return computeFletcher32(data, numBytes);
        case ChecksumType::FLETCHER64:
            return computeFletcher64(data, numBytes);
        case ChecksumType::ADLER32:
            return computeAdler32(data, numBytes);
        case ChecksumType::XXH64:
            return computeXXH64(data, numBytes);
        default:
            return 0;
    }
}

uint64_t ChecksumValidator::computeCRC32(const void* data, size_t numBytes) {
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    
    // Reset result
    uint32_t h_result = 0xFFFFFFFF;
    cudaMemcpy(d_checksum_, &h_result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(uint32_t);
    
    crc32Kernel<<<gridSize, blockSize, sharedMemSize>>>(byteData, numBytes, 
                                                        reinterpret_cast<uint32_t*>(d_checksum_));
    
    // Get result
    cudaMemcpy(&h_result, d_checksum_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return static_cast<uint64_t>(~h_result);
}

uint64_t ChecksumValidator::computeCRC64(const void* data, size_t numBytes) {
    // CPU implementation for CRC64 (GPU kernel would be similar to CRC32)
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    
    // Copy data to host if it's on device
    std::vector<uint8_t> hostData(numBytes);
    cudaMemcpy(hostData.data(), byteData, numBytes, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < numBytes; ++i) {
        crc = (crc >> 8) ^ h_crc64Table[(crc ^ hostData[i]) & 0xFF];
    }
    
    return ~crc;
}

uint64_t ChecksumValidator::computeFletcher32(const void* data, size_t numBytes) {
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    
    // Reset result
    uint32_t h_result = 0;
    cudaMemcpy(d_checksum_, &h_result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * 2 * sizeof(uint32_t);
    
    fletcher32Kernel<<<gridSize, blockSize, sharedMemSize>>>(byteData, numBytes,
                                                            reinterpret_cast<uint32_t*>(d_checksum_));
    
    // Get result
    cudaMemcpy(&h_result, d_checksum_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return static_cast<uint64_t>(h_result);
}

uint64_t ChecksumValidator::computeFletcher64(const void* data, size_t numBytes) {
    // CPU implementation for Fletcher64
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    
    // Copy data to host
    std::vector<uint8_t> hostData(numBytes);
    cudaMemcpy(hostData.data(), byteData, numBytes, cudaMemcpyDeviceToHost);
    
    uint64_t sum1 = 0;
    uint64_t sum2 = 0;
    const uint64_t MOD = 0xFFFFFFFF;
    
    for (size_t i = 0; i < numBytes; ++i) {
        sum1 = (sum1 + hostData[i]) % MOD;
        sum2 = (sum2 + sum1) % MOD;
    }
    
    return (sum2 << 32) | sum1;
}

uint64_t ChecksumValidator::computeAdler32(const void* data, size_t numBytes) {
    const uint8_t* byteData = static_cast<const uint8_t*>(data);
    
    // Reset result
    uint32_t h_result = 0;
    cudaMemcpy(d_checksum_, &h_result, sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int blockSize = 256;
    int gridSize = (numBytes + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * 2 * sizeof(uint32_t);
    
    adler32Kernel<<<gridSize, blockSize, sharedMemSize>>>(byteData, numBytes,
                                                         reinterpret_cast<uint32_t*>(d_checksum_));
    
    // Get result
    cudaMemcpy(&h_result, d_checksum_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    return static_cast<uint64_t>(h_result);
}

uint64_t ChecksumValidator::computeXXH64(const void* data, size_t numBytes) {
    // Simplified XXH64 implementation (CPU version for now)
    const uint64_t PRIME64_1 = 11400714785074694791ULL;
    const uint64_t PRIME64_2 = 14029467366897019727ULL;
    const uint64_t PRIME64_3 = 1609587929392839161ULL;
    const uint64_t PRIME64_4 = 9650029242287828579ULL;
    const uint64_t PRIME64_5 = 2870177450012600261ULL;
    
    // Copy data to host
    std::vector<uint8_t> hostData(numBytes);
    cudaMemcpy(hostData.data(), data, numBytes, cudaMemcpyDeviceToHost);
    
    uint64_t h64 = PRIME64_5 + numBytes;
    
    // Simple version - real XXH64 is more complex
    for (size_t i = 0; i < numBytes; ++i) {
        h64 ^= hostData[i] * PRIME64_5;
        h64 = ((h64 << 27) | (h64 >> 37)) * PRIME64_1 + PRIME64_4;
    }
    
    // Final mix
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;
    
    return h64;
}

bool ChecksumValidator::validateChecksum(uint64_t computed, uint64_t expected) {
    return computed == expected;
}

void ChecksumValidator::updateHistory(uint64_t checksum) {
    checksumHistory_.push_back(checksum);
    if (checksumHistory_.size() > maxHistorySize_) {
        checksumHistory_.erase(checksumHistory_.begin());
    }
}

bool ChecksumValidator::detectAnomalyInHistory() {
    if (checksumHistory_.size() < 10) {
        return false;  // Not enough history
    }
    
    // Simple anomaly detection: check if recent checksums differ significantly
    size_t recentStart = checksumHistory_.size() - 5;
    uint64_t recentChecksum = checksumHistory_[recentStart];
    
    for (size_t i = recentStart + 1; i < checksumHistory_.size(); ++i) {
        if (checksumHistory_[i] != recentChecksum) {
            return true;  // Anomaly detected
        }
    }
    
    return false;
}

void ChecksumValidator::launchCRC32Kernel(const void* data, size_t numBytes) {
    // Implementation included in computeCRC32
}

void ChecksumValidator::launchFletcher32Kernel(const void* data, size_t numBytes) {
    // Implementation included in computeFletcher32
}