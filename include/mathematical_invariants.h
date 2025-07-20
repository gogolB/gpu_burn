#ifndef MATHEMATICAL_INVARIANTS_H
#define MATHEMATICAL_INVARIANTS_H

#include "validation_engine.h"
#include <vector>
#include <unordered_map>
#include <functional>

// Types of mathematical invariants
enum class InvariantType {
    MATRIX_TRACE,           // Sum of diagonal elements
    MATRIX_DETERMINANT,     // Determinant of matrix
    MATRIX_RANK,           // Rank of matrix
    MATRIX_NORM,           // Various matrix norms
    VECTOR_MAGNITUDE,      // Vector length
    ORTHOGONALITY,         // Check if vectors are orthogonal
    SYMMETRY,              // Check matrix symmetry
    POSITIVE_DEFINITENESS, // Check if matrix is positive definite
    CONSERVATION_LAW,      // Check conservation properties
    CUSTOM                 // User-defined invariant
};

// Mathematical invariants validation
// Checks that mathematical properties are preserved during computation
class MathematicalInvariantsValidator : public ValidationMethod {
public:
    MathematicalInvariantsValidator();
    ~MathematicalInvariantsValidator() override;
    
    // ValidationMethod interface
    ValidationResult validate(
        const void* data,
        size_t numElements,
        size_t elementSize,
        const KernelConfig& config) override;
    
    std::string getName() const override { return "Mathematical Invariants"; }
    ValidationType getType() const override { return ValidationType::MATHEMATICAL_INVARIANT; }
    
    void setup(const KernelConfig& config) override;
    void cleanup() override;
    
    // Add invariant to check
    void addInvariant(InvariantType type, double expectedValue, double tolerance = 1e-6);
    
    // Custom invariant function
    using InvariantFunction = std::function<double(const void* data, size_t numElements)>;
    void addCustomInvariant(const std::string& name, InvariantFunction func, 
                           double expectedValue, double tolerance = 1e-6);
    
    // Set matrix dimensions (for matrix operations)
    void setMatrixDimensions(size_t rows, size_t cols) {
        rows_ = rows;
        cols_ = cols;
    }
    
private:
    struct InvariantCheck {
        InvariantType type;
        std::string name;
        double expectedValue;
        double tolerance;
        InvariantFunction customFunc;
        
        InvariantCheck() : type(InvariantType::CUSTOM), expectedValue(0.0), tolerance(1e-6) {}
    };
    
    std::vector<InvariantCheck> invariants_;
    size_t rows_, cols_;
    
    // Device memory for computations
    void* d_workspace_;
    size_t workspaceSize_;
    
    // Invariant computation methods
    double computeMatrixTrace(const void* data, size_t numElements, size_t elementSize);
    double computeMatrixDeterminant(const void* data, size_t numElements, size_t elementSize);
    double computeMatrixRank(const void* data, size_t numElements, size_t elementSize);
    double computeMatrixNorm(const void* data, size_t numElements, size_t elementSize, int normType);
    double computeVectorMagnitude(const void* data, size_t numElements, size_t elementSize);
    double checkOrthogonality(const void* data, size_t numElements, size_t elementSize);
    double checkSymmetry(const void* data, size_t numElements, size_t elementSize);
    double checkPositiveDefiniteness(const void* data, size_t numElements, size_t elementSize);
    double checkConservationLaw(const void* data, size_t numElements, size_t elementSize);
    
    // Helper methods
    bool validateInvariant(double computed, double expected, double tolerance);
    
    // Template methods for different types
    template<typename T>
    double computeInvariantTyped(const T* data, size_t numElements, InvariantType type);
};

// GPU kernels for invariant computations
template<typename T>
__global__ void matrixTraceKernel(const T* matrix, size_t n, T* result);

template<typename T>
__global__ void matrixNormKernel(const T* matrix, size_t rows, size_t cols, T* result, int normType);

template<typename T>
__global__ void vectorMagnitudeKernel(const T* vector, size_t n, T* result);

template<typename T>
__global__ void symmetryCheckKernel(const T* matrix, size_t n, bool* isSymmetric);

template<typename T>
__global__ void orthogonalityCheckKernel(const T* v1, const T* v2, size_t n, T* dotProduct);

// Helper functions for complex invariants
namespace MathInvariantHelpers {
    // LU decomposition for determinant computation
    template<typename T>
    void luDecomposition(T* A, size_t n, int* pivot, T* det);
    
    // Eigenvalue computation for positive definiteness
    template<typename T>
    bool computeEigenvalues(const T* matrix, size_t n, T* eigenvalues);
    
    // SVD for rank computation
    template<typename T>
    size_t computeRank(const T* matrix, size_t rows, size_t cols, T tolerance);
}

#endif // MATHEMATICAL_INVARIANTS_H