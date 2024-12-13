#include <scaluq/types.hpp>

#include "util/template.hpp"

namespace scaluq {
namespace internal {
FLOAT(Fp)
SparseMatrix<Fp>::SparseMatrix(const SparseComplexMatrix<Fp>& sp) {
    _row = sp.rows();
    _col = sp.cols();
    SparseComplexMatrix<Fp> mat = sp;
    mat.makeCompressed();

    _values = Kokkos::View<SparseValue<Fp>*>("_values", mat.nonZeros());
    Kokkos::View<SparseValue<Fp>*, Kokkos::HostSpace> values_h("values_h", mat.nonZeros());
    int idx = 0;
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (typename SparseComplexMatrix<Fp>::InnerIterator it(mat, k); it; ++it) {
            uint32_t row = it.row();
            uint32_t col = it.col();
            Complex<Fp> value = it.value();
            values_h(idx++) = {value, row, col};
        }
    }
    Kokkos::deep_copy(_values, values_h);
}
FLOAT_DECLARE_CLASS(SparseMatrix)
}  // namespace internal
}  // namespace scaluq
