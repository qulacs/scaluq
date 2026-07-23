#include <cstdint>
#include <utility>

#include "update_ops.hpp"

namespace scaluq::internal {

template <UpdatableStateVector State>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<State::prec, State::space>& matrix,
                             State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    const std::uint64_t matrix_dim = 1ULL << std::popcount(target_mask);
    State update;
    if constexpr (std::same_as<State, StateVector<State::prec, State::space>>) {
        update = State::uninitialized_state(state.n_qubits());
    } else {
        update = State::uninitialized_state(state.batch_size(), state.n_qubits());
    }

    Kokkos::parallel_for(
        "multi_dense_matrix_gate (initialize)",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update.at_unsafe(i) = 0;
            } else {
                update.at_unsafe(i) = state.at_unsafe(i);
            }
        });

    const std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        "multi_dense_matrix_gate (update)",
        Kokkos::TeamPolicy<ExecSpace>(ExecSpace(),
                                      state.flat_dim() >> std::popcount(target_mask | control_mask),
                                      Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t r) {
                const std::uint64_t dst_index =
                    insert_zero_at_mask_positions(r, outer_mask) | basis;
                ComplexType sum = Float<State::prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t c, ComplexType& inner_sum) {
                        const std::uint64_t src_index =
                            insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += matrix(r, c) * state.at_unsafe(src_index);
                    },
                    sum);
                update.at_unsafe(dst_index) = sum;
            });
            team.team_barrier();
        });

    std::swap(state._raw, update._raw);
}

template <UpdatableStateVector State>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<State::prec, State::space>& mat,
                        State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    State update;
    if constexpr (std::same_as<State, StateVector<State::prec, State::space>>) {
        update = State::uninitialized_state(state.n_qubits());
    } else {
        update = State::uninitialized_state(state.batch_size(), state.n_qubits());
    }

    Kokkos::parallel_for(
        "sparse_matrix_gate (initialize)",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim()),
        KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i & control_mask) == control_value_mask) {
                update.at_unsafe(i) = 0;
            } else {
                update.at_unsafe(i) = state.at_unsafe(i);
            }
        });

    const std::uint64_t outer_mask = ~target_mask & ((1ULL << state.n_qubits()) - 1);
    Kokkos::parallel_for(
        "sparse_matrix_gate (update)",
        Kokkos::TeamPolicy<ExecSpace>(ExecSpace(),
                                      state.flat_dim() >> std::popcount(target_mask | control_mask),
                                      Kokkos::AUTO),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) {
            std::uint64_t basis = team.league_rank();
            basis = insert_zero_at_mask_positions(basis, target_mask | control_mask) |
                    control_value_mask;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, mat._rows), [&](std::uint64_t r) {
                const std::uint64_t dst_index =
                    insert_zero_at_mask_positions(r, outer_mask) | basis;
                ComplexType sum = Float<State::prec>{0};
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, mat._row_ptr[r], mat._row_ptr[r + 1]),
                    [&](std::uint64_t idx, ComplexType& inner_sum) {
                        const std::uint64_t c = mat._col_idx[idx];
                        const std::uint64_t src_index =
                            insert_zero_at_mask_positions(c, outer_mask) | basis;
                        inner_sum += mat._vals[idx] * state.at_unsafe(src_index);
                    },
                    sum);
                update.at_unsafe(dst_index) = sum;
            });
            team.team_barrier();
        });

    std::swap(state._raw, update._raw);
}

#define INSTANTIATE_FLAT_STATE_OVERLOADS(Func, ...)                                      \
    template void Func<StateVector<Prec, Space>>(__VA_ARGS__, StateVector<Prec, Space>&); \
    template void Func<StateVectorBatched<Prec, Space>>(__VA_ARGS__,                      \
                                                        StateVectorBatched<Prec, Space>&)
// clang-format off
INSTANTIATE_FLAT_STATE_OVERLOADS(multi_dense_matrix_gate,       std::uint64_t, std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(sparse_matrix_gate,            std::uint64_t, std::uint64_t, std::uint64_t, const SparseMatrix<Prec, Space>&);

#undef INSTANTIATE_FLAT_STATE_OVERLOADS
// clang-format on

}  // namespace scaluq::internal
