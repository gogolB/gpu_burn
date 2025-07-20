#ifndef TMR_VALIDATOR_H
#define TMR_VALIDATOR_H

#include "validation_engine.h"
#include <vector>
#include <functional>

// TMR voting strategies
enum class VotingStrategy {
    MAJORITY,      // Simple majority voting (2 out of 3 agree)
    UNANIMOUS,     // All 3 must agree
    WEIGHTED,      // Weighted voting based on confidence
    BITWISE        // Bitwise majority voting for each bit
};

// TMR execution modes
enum class TMRMode {
    SEQUENTIAL,    // Run 3 times sequentially
    CONCURRENT,    // Run 3 kernels concurrently on same GPU
    MULTI_GPU      // Run on 3 different GPUs if available
};

// Triple Modular Redundancy validator
// Executes computation 3 times and votes on results
class TMRValidator : public ValidationMethod {
public:
    TMRValidator(VotingStrategy strategy = VotingStrategy::MAJORITY,
                 TMRMode mode = TMRMode::SEQUENTIAL);
    ~TMRValidator() override;
    
    // ValidationMethod interface
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override;
    
    std::string getName() const override { return "Triple Modular Redundancy"; }
    ValidationType getType() const override { return ValidationType::TMR; }
    
    void setup(const KernelConfig& config) override;
    void cleanup() override;
    
    // Set kernel execution function
    using KernelFunction = std::function<void(void* output, const void* input, 
                                             size_t numElements, const KernelConfig& config)>;
    void setKernelFunction(KernelFunction func) { kernelFunc_ = func; }
    
    // Set voting parameters
    void setVotingStrategy(VotingStrategy strategy) { votingStrategy_ = strategy; }
    void setTMRMode(TMRMode mode) { tmrMode_ = mode; }
    
    // Enable detailed diagnostics
    void enableDiagnostics(bool enable) { diagnosticsEnabled_ = enable; }
    
    // Get execution statistics
    struct TMRStats {
        size_t totalRuns;
        size_t consensusFailures;
        size_t singleBitErrors;
        size_t multiBitErrors;
        double averageAgreementRate;
        
        TMRStats() : totalRuns(0), consensusFailures(0), 
                    singleBitErrors(0), multiBitErrors(0), 
                    averageAgreementRate(1.0) {}
    };
    
    TMRStats getStats() const { return stats_; }
    
private:
    VotingStrategy votingStrategy_;
    TMRMode tmrMode_;
    KernelFunction kernelFunc_;
    bool diagnosticsEnabled_;
    TMRStats stats_;
    
    // Device buffers for TMR execution
    void* d_output1_;
    void* d_output2_;
    void* d_output3_;
    void* d_votedOutput_;
    size_t bufferSize_;
    
    // Voting functions
    ValidationResult performMajorityVoting(const void* out1, const void* out2, 
                                         const void* out3, void* votedOutput,
                                         size_t numElements, size_t elementSize);
    
    ValidationResult performUnanimousVoting(const void* out1, const void* out2,
                                          const void* out3, void* votedOutput,
                                          size_t numElements, size_t elementSize);
    
    ValidationResult performWeightedVoting(const void* out1, const void* out2,
                                         const void* out3, void* votedOutput,
                                         size_t numElements, size_t elementSize);
    
    ValidationResult performBitwiseVoting(const void* out1, const void* out2,
                                        const void* out3, void* votedOutput,
                                        size_t numElements, size_t elementSize);
    
    // Helper functions
    void executeSequential(const void* input, size_t numElements, const KernelConfig& config);
    void executeConcurrent(const void* input, size_t numElements, const KernelConfig& config);
    void executeMultiGPU(const void* input, size_t numElements, const KernelConfig& config);
    
    // Comparison functions
    bool compareOutputs(const void* out1, const void* out2, size_t numBytes);
    size_t countDifferences(const void* out1, const void* out2, size_t numBytes);
    
    // Diagnostic functions
    void analyzeDifferences(const void* out1, const void* out2, const void* out3,
                          size_t numElements, size_t elementSize);
};

// GPU kernels for TMR voting
__global__ void majorityVoteKernel(const uint8_t* in1, const uint8_t* in2, 
                                  const uint8_t* in3, uint8_t* out, size_t numBytes);

__global__ void bitwiseVoteKernel(const uint8_t* in1, const uint8_t* in2,
                                 const uint8_t* in3, uint8_t* out, size_t numBytes);

__global__ void compareAndCountKernel(const uint8_t* data1, const uint8_t* data2,
                                    size_t numBytes, uint32_t* diffCount);

// Helper class for concurrent TMR execution
class ConcurrentTMRExecutor {
public:
    ConcurrentTMRExecutor();
    ~ConcurrentTMRExecutor();
    
    void execute(void* out1, void* out2, void* out3,
                const void* input, size_t numElements,
                const TMRValidator::KernelFunction& kernelFunc,
                const KernelConfig& config);
    
private:
    cudaStream_t stream1_, stream2_, stream3_;
};

#endif // TMR_VALIDATOR_H