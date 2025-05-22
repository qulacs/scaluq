#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& mat,
                        StateVector<Prec, Space>& state) {
    auto values = mat._values;

    Kokkos::View<Complex<Prec>*, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        "sparse_matrix_gate (initialize)",
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::View<Complex<Prec>*, SpaceType<Space>, Kokkos::MemoryTraits<Kokkos::Atomic>>
        update_atomic(update);
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0},
            {static_cast<std::int64_t>(state.dim() >> std::popcount(target_mask | control_mask)),
             static_cast<std::int64_t>(values.size())}),
        KOKKOS_LAMBDA(std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_value_mask;
            auto [v, r, c] = values(inner);
            std::uint32_t src_index =
                internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            std::uint32_t dst_index =
                internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    state._raw = update;
}

template <>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& mat,
                        StateVectorBatched<Prec, Space>& states) {
    auto values = mat._values;
    const std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);

    Kokkos::View<Complex<Prec>**, Kokkos::LayoutRight, SpaceType<Space>> update(
        Kokkos::ViewAllocateWithoutInitializing("update"), states.batch_size(), states.dim());

    Kokkos::parallel_for(
        "sparse_matrix_gate (initialize)",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);
            }
        });
    Kokkos::fence();

    Kokkos::View<Complex<Prec>**,
                 Kokkos::LayoutRight,
                 SpaceType<Space>,
                 Kokkos::MemoryTraits<Kokkos::Atomic>>
        update_atomic(update);
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<3>>(
            {0, 0, 0},
            {static_cast<std::int64_t>(states.batch_size()),
             static_cast<std::int64_t>(states.dim() >> std::popcount(target_mask | control_mask)),
             static_cast<std::int64_t>(values.size())}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_value_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            update_atomic(batch_id, dst_index) += v * states._raw(batch_id, src_index);
        });
    Kokkos::fence();
    states._raw = update;
}

}  // namespace scaluq::internal
