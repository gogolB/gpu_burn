#ifndef CHECKSUM_VALIDATOR_H
#define CHECKSUM_VALIDATOR_H

#include "validation_engine.h"
#include <vector>

// Checksum types supported
enum class ChecksumType {
    CRC32,
    CRC64,
    FLETCHER32,
    FLETCHER64,
    ADLER32,
    XXH64  // xxHash for high-speed hashing
};

// Checksum validation method
// Computes and verifies checksums of GPU computation results
class ChecksumValidator : public ValidationMethod {
public:
    ChecksumValidator(ChecksumType type = ChecksumType::CRC32);
    ~ChecksumValidator() override;
    
    // ValidationMethod interface
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override;
    
    std::string getName() const override { return "Checksum Validator"; }
    ValidationType getType() const override { return ValidationType::CHECKSUM; }
    
    void setup(const KernelConfig& config) override;
    void cleanup() override;
    
    // Set expected checksum (for pre-computed values)
    void setExpectedChecksum(uint64_t checksum) { expectedChecksum_ = checksum; hasExpected_ = true; }
    
    // Compute checksum without validation (for generating expected values)
    uint64_t computeChecksum(const void* data, size_t numBytes);
    
    // Enable continuous checksum tracking
    void enableContinuousMode(bool enable) { continuousMode_ = enable; }
    
private:
    ChecksumType checksumType_;
    uint64_t expectedChecksum_;
    bool hasExpected_;
    bool continuousMode_;
    
    // History for continuous mode
    std::vector<uint64_t> checksumHistory_;
    size_t maxHistorySize_;
    
    // GPU checksum computation
    uint64_t* d_checksum_;
    void* d_scratchBuffer_;
    size_t scratchBufferSize_;
    
    // Checksum computation methods
    uint64_t computeCRC32(const void* data, size_t numBytes);
    uint64_t computeCRC64(const void* data, size_t numBytes);
    uint64_t computeFletcher32(const void* data, size_t numBytes);
    uint64_t computeFletcher64(const void* data, size_t numBytes);
    uint64_t computeAdler32(const void* data, size_t numBytes);
    uint64_t computeXXH64(const void* data, size_t numBytes);
    
    // GPU kernels for checksum computation
    void launchCRC32Kernel(const void* data, size_t numBytes);
    void launchFletcher32Kernel(const void* data, size_t numBytes);
    
    // Validation helpers
    bool validateChecksum(uint64_t computed, uint64_t expected);
    void updateHistory(uint64_t checksum);
    bool detectAnomalyInHistory();
};

// GPU kernels for checksum computation
__global__ void crc32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result);
__global__ void crc64Kernel(const uint8_t* data, size_t numBytes, uint64_t* result);
__global__ void fletcher32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result);
__global__ void fletcher64Kernel(const uint8_t* data, size_t numBytes, uint64_t* result);
__global__ void adler32Kernel(const uint8_t* data, size_t numBytes, uint32_t* result);

// CRC lookup tables (device constant memory)
extern __constant__ uint32_t d_crc32Table[256];
extern __constant__ uint64_t d_crc64Table[256];

#endif // CHECKSUM_VALIDATOR_H