#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace scaluq {

using InitializationSettings = Kokkos::InitializationSettings;

inline void initialize(const InitializationSettings& settings = InitializationSettings()) {
    Kokkos::initialize(settings);
}
inline void finalize() { Kokkos::finalize(); }
inline bool is_initialized() { return Kokkos::is_initialized(); }
inline bool is_finalized() { return Kokkos::is_finalized(); }

using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;

using StdComplex = std::complex<double>;
using ComplexMatrix = Eigen::Matrix<StdComplex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<StdComplex, Eigen::RowMajor>;

using Matrix = Kokkos::View<Complex**, Kokkos::LayoutRight>;

using StateVectorView = Kokkos::View<Complex*>;
using StateVectorBatchedView = Kokkos::View<Complex**, Kokkos::LayoutRight>;

struct array_4 {
    Complex val[4];
};

struct matrix_2_2 {
    Complex val[2][2];
};

struct matrix_4_4 {
    Complex val[4][4];
};

struct diagonal_matrix_2_2 {
    Complex val[2];
};

struct SparseValue {
    Complex val;
    uint32_t r, c;
};

class SparseMatrix {
public:
    Kokkos::View<SparseValue*> _values;
    UINT _row, _col;

    SparseMatrix(const SparseComplexMatrix& sp) {
        _row = sp.rows();
        _col = sp.cols();
        SparseComplexMatrix mat = sp;
        mat.makeCompressed();

        Kokkos::View<SparseValue*, Kokkos::HostSpace> values_h("values_h", mat.nonZeros());
        int idx = 0;
        for (int k = 0; k < mat.outerSize(); ++k) {
            for (SparseComplexMatrix::InnerIterator it(mat, k); it; ++it) {
                uint32_t row = it.row();
                uint32_t col = it.col();
                Complex value = it.value();
                values_h(idx++) = {value, row, col};
            }
        }
        Kokkos::deep_copy(_values, values_h);
    }
};

}  // namespace scaluq
