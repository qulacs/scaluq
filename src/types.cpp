#include <scaluq/types.hpp>

#include "util/template.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
SparseMatrix<Prec, Space>::SparseMatrix(const SparseComplexMatrix& sp) {
    _row = sp.rows();
    _col = sp.cols();
    SparseComplexMatrix mat = sp;
    mat.makeCompressed();

    _values = Kokkos::View<SparseValue<Prec>*, SpaceType<Space>>("_values", mat.nonZeros());
    Kokkos::View<SparseValue<Prec>*, Kokkos::HostSpace> values_h("values_h", mat.nonZeros());
    int idx = 0;
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (typename SparseComplexMatrix::InnerIterator it(mat, k); it; ++it) {
            uint32_t row = it.row();
            uint32_t col = it.col();
            Complex<Prec> value = it.value();
            values_h(idx++) = {value, row, col};
        }
    }
    Kokkos::deep_copy(_values, values_h);
}
SCALUQ_DECLARE_CLASS_FOR_PRECISION_AND_EXECUTION_SPACE(SparseMatrix)
}  // namespace internal
}  // namespace scaluq
