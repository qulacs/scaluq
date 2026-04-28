#pragma once

#include <complex>
#include <vector>

#ifdef SCALUQ_STATIC_ANALYSIS

// NOLINTBEGIN

namespace Eigen {

constexpr int Dynamic = -1;
constexpr int RowMajor = 1;

template <class Scalar>
struct Triplet {
    Triplet() = default;
    Triplet(int, int, const Scalar& = {}) {}
    int row() const { return 0; }
    int col() const { return 0; }
    Scalar value() const { return {}; }
};

template <class Scalar, int Rows = Dynamic, int Cols = Dynamic, int Options = 0>
class Matrix {
public:
    Matrix() = default;
    Matrix(int, int) {}
    int rows() const { return 0; }
    int cols() const { return 0; }
    void resize(int, int) {}
    void setZero() {}
    Scalar& operator()(int, int) {
        static Scalar v{};
        return v;
    }
    const Scalar& operator()(int, int) const {
        static Scalar v{};
        return v;
    }
    Scalar* data() { return nullptr; }
    const Scalar* data() const { return nullptr; }
    static Matrix Zero(int, int) { return {}; }
    static Matrix Identity(int, int) { return {}; }
    Matrix adjoint() const { return {}; }
    Matrix eval() const { return {}; }
    Matrix operator*(const Matrix&) const { return {}; }
    Matrix operator*(const Scalar&) const { return {}; }
    Matrix& operator/=(double) { return *this; }
    Matrix& operator/=(const std::complex<double>&) { return *this; }
    struct CommaInit {
        template <class T>
        CommaInit& operator,(T) {
            return *this;
        }
    };
    template <class T>
    CommaInit operator<<(T) {
        return {};
    }
    struct Block {
        Block& operator=(const Matrix&) { return *this; }
    };
    Block block(int, int, int, int) { return {}; }
    Block block(int, int, int, int) const { return {}; }
};

template <class Scalar, int R, int C, int O>
inline Matrix<Scalar, R, C, O> operator*(const Scalar&, const Matrix<Scalar, R, C, O>&) {
    return {};
}

enum class ComputationInfo { Success, NumericalIssue, NoConvergence, InvalidInput };

template <class MatrixType>
class ComplexEigenSolver {
public:
    using Scalar = typename MatrixType::Scalar;
    using EigenvalueType = std::vector<Scalar>;
    explicit ComplexEigenSolver(const MatrixType&) {}
    const EigenvalueType& eigenvalues() const {
        static EigenvalueType v;
        return v;
    }
    const MatrixType& eigenvectors() const {
        static MatrixType m;
        return m;
    }
    ComputationInfo info() const { return ComputationInfo::Success; }
};

}  // namespace Eigen

namespace scaluq {

class SparseComplexMatrix;

class ComplexMatrix {
public:
    using Scalar = std::complex<double>;

    ComplexMatrix() = default;
    ComplexMatrix(int, int) {}

    int rows() const { return 0; }
    int cols() const { return 0; }
    void resize(int, int) {}
    void setZero() {}

    Scalar& operator()(int, int) {
        static Scalar v{};
        return v;
    }
    const Scalar& operator()(int, int) const {
        static Scalar v{};
        return v;
    }
    Scalar* data() { return nullptr; }
    const Scalar* data() const { return nullptr; }

    struct Block {
        Block& operator=(const ComplexMatrix&) { return *this; }
    };
    Block block(int, int, int, int) { return {}; }
    Block block(int, int, int, int) const { return {}; }

    SparseComplexMatrix sparseView() const;

    static ComplexMatrix Zero(int, int) { return {}; }
    static ComplexMatrix Identity(int, int) { return {}; }
    ComplexMatrix adjoint() const { return {}; }
    ComplexMatrix eval() const { return {}; }

    ComplexMatrix operator+(const ComplexMatrix&) const { return {}; }
    ComplexMatrix operator-(const ComplexMatrix&) const { return {}; }
    ComplexMatrix operator*(const ComplexMatrix&) const { return {}; }
    ComplexMatrix operator*(const Scalar&) const { return {}; }
    ComplexMatrix& operator/=(double) { return *this; }
    ComplexMatrix& operator/=(const Scalar&) { return *this; }

    struct CommaInit {
        template <class T>
        CommaInit& operator,(T) {
            return *this;
        }
    };
    template <class T>
    CommaInit operator<<(T) {
        return {};
    }
};

inline ComplexMatrix operator*(std::complex<double>, const ComplexMatrix&) { return {}; }

class SparseComplexMatrix {
public:
    struct InnerIterator {
        InnerIterator(const SparseComplexMatrix&, int) {}
        explicit operator bool() const { return false; }
        InnerIterator& operator++() { return *this; }
        int row() const { return 0; }
        int col() const { return 0; }
        std::complex<double> value() const { return {}; }
    };

    SparseComplexMatrix() = default;
    SparseComplexMatrix(int, int) {}

    int rows() const { return 0; }
    int cols() const { return 0; }
    int outerSize() const { return 0; }
    int nonZeros() const { return 0; }

    void resize(int, int) {}
    template <class It>
    void setFromTriplets(It, It) {}

    ComplexMatrix toDense() const { return {}; }

    std::complex<double>* valuePtr() { return nullptr; }
    const std::complex<double>* valuePtr() const { return nullptr; }
    int* innerIndexPtr() { return nullptr; }
    const int* innerIndexPtr() const { return nullptr; }
    int* outerIndexPtr() { return nullptr; }
    const int* outerIndexPtr() const { return nullptr; }
};

inline SparseComplexMatrix ComplexMatrix::sparseView() const { return {}; }

}  // namespace scaluq

// NOLINTEND

#else  // ── Real Eigen ──────────────────────────────────────────────────────────

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace scaluq {
using ComplexMatrix =
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
}  // namespace scaluq

#endif
