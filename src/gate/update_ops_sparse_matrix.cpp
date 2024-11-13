#include "../util/template.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVector<Fp>& state) {
    auto values = mat._values;

    Kokkos::View<Complex<Fp>*> update(Kokkos::ViewAllocateWithoutInitializing("update"),
                                      state.dim());
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::View<Complex<Fp>*, Kokkos::MemoryTraits<Kokkos::Atomic>> update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0},
            {static_cast<std::int64_t>(state.dim() >> std::popcount(target_mask | control_mask)),
             static_cast<std::int64_t>(values.size())}),
        KOKKOS_LAMBDA(std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    state._raw = update;
}
#define FUNC_MACRO(Fp)                \
    template void sparse_matrix_gate( \
        std::uint64_t, std::uint64_t, const SparseMatrix<Fp>&, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
