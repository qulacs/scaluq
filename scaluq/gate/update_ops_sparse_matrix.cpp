#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix& mat,
                        StateVector& state) {
    auto values = mat._values;

    Kokkos::View<Complex*> update(Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
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
    Kokkos::View<Complex*, Kokkos::MemoryTraits<Kokkos::Atomic>> update_atomic(update);
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

void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix& mat,
                        StateVectorBatched& states) {
    auto values = mat._values;
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    const std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);

    Kokkos::View<Complex**, Kokkos::LayoutRight> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), states.batch_size(), states.dim());

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);
            }
        });
    Kokkos::fence();

    Kokkos::View<Complex**, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Atomic>>
        update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0},
            {states.batch_size(),
             states.dim() >> std::popcount(target_mask | control_mask),
             values.size()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(batch_id, dst_index) += v * states._raw(batch_id, src_index);
        });
    Kokkos::fence();
    states._raw = update;
}

}  // namespace internal
}  // namespace scaluq
