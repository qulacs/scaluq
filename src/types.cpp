#include <scaluq/types.hpp>

#include "util/template.hpp"

namespace scaluq {
namespace internal {
FLOAT_AND_SPACE(Fp, Sp)
SparseMatrix<Fp, Sp>::SparseMatrix(const SparseComplexMatrix<Fp>& sp)
    : _row(sp.rows()), _col(sp.cols()), _values("_values", sp.nonZeros()) {
    Kokkos::View<SparseValue<Fp>*, Kokkos::HostSpace> values_h("values_h", sp.nonZeros());
    int idx = 0;
    for (int k = 0; k < sp.outerSize(); ++k) {
        for (typename SparseComplexMatrix<Fp>::InnerIterator it(sp, k); it; ++it) {
            uint32_t row = it.row();
            uint32_t col = it.col();
            Complex<Fp> value = it.value();
            values_h(idx++) = {value, row, col};
        }
    }
    Kokkos::deep_copy(_values, values_h);
}
FLOAT_AND_SPACE_DECLARE_CLASS(SparseMatrix)
}  // namespace internal
}  // namespace scaluq
