#include <scaluq/types.hpp>

#include "prec_space.hpp"

namespace scaluq {
namespace internal {
template <Precision Prec, ExecutionSpace Space>
SparseMatrix<Prec, Space>::SparseMatrix(const SparseComplexMatrix& sp)
    : _vals(Kokkos::ViewAllocateWithoutInitializing("_vals"), sp.nonZeros()),
      _col_idx(Kokkos::ViewAllocateWithoutInitializing("_col_idx"), sp.nonZeros()),
      _row_ptr(Kokkos::ViewAllocateWithoutInitializing("_row_ptr"), sp.rows() + 1),
      _rows(sp.rows()),
      _cols(sp.cols()) {
    Kokkos::View<ComplexType*, Kokkos::HostSpace> vals_h("vals_h", sp.nonZeros());
    std::copy(sp.valuePtr(), sp.valuePtr() + sp.nonZeros(), vals_h.data());
    Kokkos::View<std::uint32_t*, Kokkos::HostSpace> col_idx_h("col_idx_h", sp.nonZeros()),
        row_ptr_h("row_ptr_h", sp.rows() + 1);
    std::copy(sp.innerIndexPtr(), sp.innerIndexPtr() + sp.nonZeros(), col_idx_h.data());
    std::copy(sp.outerIndexPtr(), sp.outerIndexPtr() + sp.rows() + 1, row_ptr_h.data());
    Kokkos::deep_copy(_vals, vals_h);
    Kokkos::deep_copy(_col_idx, col_idx_h);
    Kokkos::deep_copy(_row_ptr, row_ptr_h);
}
template class SparseMatrix<Prec, Space>;
}  // namespace internal
}  // namespace scaluq
