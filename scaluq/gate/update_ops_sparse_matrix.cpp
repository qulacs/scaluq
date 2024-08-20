#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void sparse_matrix_gate(UINT target_mask,
                        UINT control_mask,
                        const SparseMatrix& mat,
                        StateVector& state) {
    using Value = Kokkos::pair<Complex, uint32_t>;
    UINT full_mask = (1ULL << state.n_qubits()) - 1;
    UINT inner_mask = ~(target_mask | control_mask) & full_mask;
    auto values = mat._values;
    Kokkos::View<Complex*> update(Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(UINT i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    Kokkos::Experimental::ScatterView<Complex*> scatter(update);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {state.dim() >> std::popcount(target_mask | control_mask), values.size()}),
        KOKKOS_LAMBDA(UINT it, UINT i) {
            UINT basis = internal::insert_zero_at_mask_positions(it, control_mask | target_mask) |
                         control_mask;
            auto access = scatter.access();
            auto [v, r, c] = values(i);
            uint32_t src_index = internal::insert_zero_at_mask_positions(r, inner_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(c, inner_mask) | basis;
            access(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    Kokkos::Experimental::contribute(update, scatter);
    state._raw = update;
}
}  // namespace internal
}  // namespace scaluq
