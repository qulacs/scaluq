#include <algorithm>

#include "update_ops.hpp"

namespace scaluq::internal {
template <>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<Prec, Space>& matrix,
                             StateVector<Prec, Space>& state) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    typename StateVector<Prec, Space>::RawView update(
        Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        "multi_dense_matrix_gate (initialize)",
        Kokkos::RangePolicy<SpaceType<Space>>(0, state.dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(i) = -1;
            } else {
                update(i) = state._raw(i);
            }
        });

    std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        "multi_dense_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(),
            state.dim() >> std::popcount(target_mask | control_mask),
            Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex<Prec>& inner_sum) {
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += matrix(r, c) * state._raw(src_index);
                    },
                    sum);
                update(dst_index) = sum;
            });
            team.team_barrier();
        });

    state._raw = update;
}

template <>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<Prec, Space>& matrix,
                             StateVectorBatched<Prec, Space>& states) {
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);

    typename StateVectorBatched<Prec, Space>::RawView update(
        Kokkos::ViewAllocateWithoutInitializing("update"),
        Kokkos::LayoutStride(
            states.batch_size(), std::max(states.dim(), std::uint64_t{8}), states.dim(), 1));

    Kokkos::parallel_for(
        "multi_dense_matrix_gate (initialize)",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update(batch_id, i) = 0;
            } else {
                update(batch_id, i) = states._raw(batch_id, i);  // 制御条件を満たさないインデクス
            }
        });

    std::uint64_t outer_size = states.dim() >> std::popcount(target_mask | control_mask);
    std::uint64_t outer_mask = ~target_mask & ((1ULL << states.n_qubits()) - 1);
    Kokkos::parallel_for(
        "multi_dense_matrix_gate (update)",
        Kokkos::TeamPolicy<SpaceType<Space>>(
            SpaceType<Space>(), outer_size * states.batch_size(), Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            std::uint64_t basis = team.league_rank() % outer_size;
            std::uint64_t batch_id = team.league_rank() / outer_size;
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                std::uint64_t dst_index =
                    internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
                Complex<Prec> sum = Float<Prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, Complex<Prec>& inner_sum) {
                        std::uint64_t src_index =
                            internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += matrix(r, c) * states._raw(batch_id, src_index);
                    },
                    sum);
                update(batch_id, dst_index) = sum;
            });
            team.team_barrier();
        });

    states._raw = update;
}
}  // namespace scaluq::internal
