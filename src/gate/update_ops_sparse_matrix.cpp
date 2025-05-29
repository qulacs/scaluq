#include "../prec_space.hpp"
#include "update_ops.hpp"

namespace scaluq::internal {
template <>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& mat,
                        StateVector<Prec, Space>& state) {
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
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(),
            state.dim() >> std::popcount(target_mask | control_mask),
            Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mat._rows), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, mat._row_ptr[r], mat._row_ptr[r + 1]),
                    [&](std::uint64_t idx, Complex<Prec>& inner_sum) {
                        std::uint64_t c = mat._col_idx[idx];
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += mat._vals[idx] * state._raw(src_index);
                    },
                    sum);
                update(dst_index) = sum;
            });
            team.team_barrier();
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

    std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);
    std::uint64_t outer_size = states.dim() >> std::popcount(target_mask | control_mask);
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(), outer_size * states.batch_size(), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank() % outer_size;
            std::uint64_t batch_id = team.league_rank() / outer_size;
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mat._rows), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, mat._row_ptr[r], mat._row_ptr[r + 1]),
                    [&](std::uint64_t idx, Complex<Prec>& inner_sum) {
                        std::uint64_t c = mat._col_idx[idx];
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += mat._vals[idx] * states._raw(batch_id, src_index);
                    },
                    sum);
                update(batch_id, dst_index) = sum;
            });
            team.team_barrier();
        });
    Kokkos::fence();

    states._raw = update;
}

}  // namespace scaluq::internal
