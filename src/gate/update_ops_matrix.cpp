#include <cstdint>
#include <scaluq/util/simd_complex.hpp>
#include <utility>

#include "update_ops.hpp"

namespace scaluq::internal {

template <UpdatableStateVector State>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<State::prec>& diag,
                                     State& state) {
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "one_target_diagonal_matrix_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            state.at_unsafe(basis) *= diag[0];
            state.at_unsafe(basis | target_mask) *= diag[1];
        });
}

template <UpdatableStateVector State>
void zero_target_dense_matrix_gate_simd(std::uint64_t control_mask,
                                        std::uint64_t control_value_mask,
                                        Complex<State::prec> matrix,
                                        State& state) {
    using SimdComplex = internal::SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdComplex::complex_lanes;
    const auto coef = SimdComplex::Coef::splat(matrix);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(control_mask);
    const std::uint64_t total_work_items = flat_span / complex_lanes;
    Kokkos::parallel_for(
        "zero_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, total_work_items),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, control_mask) | control_value_mask;
            auto v = SimdComplex::load_aligned(&state.at_unsafe(basis));
            (coef * v).store_aligned(&state.at_unsafe(basis));
        });
}

template <UpdatableStateVector State>
void zero_target_dense_matrix_gate_scalar(std::uint64_t control_mask,
                                          std::uint64_t control_value_mask,
                                          Complex<State::prec> matrix,
                                          State& state) {
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "zero_target_dense_matrix_gate",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
            state.at_unsafe(basis) *= matrix;
        });
}

template <UpdatableStateVector State>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<State::prec> matrix,
                                   State& state) {
    if constexpr ((State::space == ExecutionSpace::Host ||
                   State::space == ExecutionSpace::HostSerial) &&
                  (State::prec == Precision::F64 || State::prec == Precision::F32)) {
        constexpr std::size_t complex_lanes = internal::SimdComplex<State::prec>::complex_lanes;
        if constexpr (complex_lanes > 0) {
            const std::uint64_t span = state.dim() >> std::popcount(control_mask);
            if (span >= complex_lanes && (control_mask & (complex_lanes - 1)) == 0) {
                zero_target_dense_matrix_gate_simd(control_mask, control_value_mask, matrix, state);
                return;
            }
        }
    }
    zero_target_dense_matrix_gate_scalar(control_mask, control_value_mask, matrix, state);
}

template <UpdatableStateVector State>
void one_target_dense_matrix_gate_simd(std::uint64_t target_mask,
                                       std::uint64_t control_mask,
                                       std::uint64_t control_value_mask,
                                       const Matrix2x2<State::prec>& matrix,
                                       State& state) {
    using SimdComplex = internal::SimdComplex<State::prec>;
    using Coef = typename SimdComplex::Coef;
    constexpr std::size_t complex_lanes = SimdComplex::complex_lanes;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const Coef m00 = Coef::splat(matrix[0][0]);
    const Coef m01 = Coef::splat(matrix[0][1]);
    const Coef m10 = Coef::splat(matrix[1][0]);
    const Coef m11 = Coef::splat(matrix[1][1]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    const std::uint64_t total_work_items = flat_span / complex_lanes;
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, total_work_items),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target_mask;
            const auto v0 = SimdComplex::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = SimdComplex::load_aligned(&state.at_unsafe(basis1));
            (m00 * v0 + m01 * v1).store_aligned(&state.at_unsafe(basis0));
            (m10 * v0 + m11 * v1).store_aligned(&state.at_unsafe(basis1));
        });
}

template <UpdatableStateVector State>
void one_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                         std::uint64_t control_mask,
                                         std::uint64_t control_value_mask,
                                         const Matrix2x2<State::prec>& matrix,
                                         State& state) {
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            std::uint64_t basis_1 = basis_0 | target_mask;
            Complex<State::prec> val0 = state.at_unsafe(basis_0);
            Complex<State::prec> val1 = state.at_unsafe(basis_1);
            Complex<State::prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
            Complex<State::prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
            state.at_unsafe(basis_0) = res0;
            state.at_unsafe(basis_1) = res1;
        });
}

template <UpdatableStateVector State>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<State::prec>& matrix,
                                  State& state) {
    if constexpr ((State::space == ExecutionSpace::Host ||
                   State::space == ExecutionSpace::HostSerial) &&
                  (State::prec == Precision::F64 || State::prec == Precision::F32)) {
        constexpr std::size_t complex_lanes = internal::SimdComplex<State::prec>::complex_lanes;
        if constexpr (complex_lanes > 0) {
            const std::uint64_t skip_mask = target_mask | control_mask;
            const std::uint64_t span = state.dim() >> std::popcount(skip_mask);
            if (span >= complex_lanes && (skip_mask & (complex_lanes - 1)) == 0) {
                // TODO: (skip_mask & (complex_lanes - 1)) != 0 の場合についてもSIMDを使うようにする
                one_target_dense_matrix_gate_simd(
                    target_mask, control_mask, control_value_mask, matrix, state);
                return;
            }
        }
    }
    one_target_dense_matrix_gate_scalar(
        target_mask, control_mask, control_value_mask, matrix, state);
}

template <UpdatableStateVector State>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<State::prec>& matrix,
                                  State& state) {
    using ExecSpace = SpaceType<State::space>;
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex<State::prec> val0 = state.at_unsafe(basis_0);
            Complex<State::prec> val1 = state.at_unsafe(basis_1);
            Complex<State::prec> val2 = state.at_unsafe(basis_2);
            Complex<State::prec> val3 = state.at_unsafe(basis_3);
            Complex<State::prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1 +
                                        matrix[0][2] * val2 + matrix[0][3] * val3;
            Complex<State::prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1 +
                                        matrix[1][2] * val2 + matrix[1][3] * val3;
            Complex<State::prec> res2 = matrix[2][0] * val0 + matrix[2][1] * val1 +
                                        matrix[2][2] * val2 + matrix[2][3] * val3;
            Complex<State::prec> res3 = matrix[3][0] * val0 + matrix[3][1] * val1 +
                                        matrix[3][2] * val2 + matrix[3][3] * val3;
            state.at_unsafe(basis_0) = res0;
            state.at_unsafe(basis_1) = res1;
            state.at_unsafe(basis_2) = res2;
            state.at_unsafe(basis_3) = res3;
        });
}

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

// clang-format off
#define INSTANTIATE_FLAT_STATE_OVERLOADS(Func, ...)                                      \
    template void Func<StateVector<Prec, Space>>(__VA_ARGS__, StateVector<Prec, Space>&); \
    template void Func<StateVectorBatched<Prec, Space>>(                                 \
        __VA_ARGS__, StateVectorBatched<Prec, Space>&)

INSTANTIATE_FLAT_STATE_OVERLOADS(zero_target_dense_matrix_gate, std::uint64_t, std::uint64_t, Complex<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(one_target_dense_matrix_gate,  std::uint64_t, std::uint64_t, std::uint64_t, const Matrix2x2<Prec>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(two_target_dense_matrix_gate,  std::uint64_t, std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(one_target_diagonal_matrix_gate, std::uint64_t, std::uint64_t, std::uint64_t, const DiagonalMatrix2x2<Prec>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(multi_dense_matrix_gate,       std::uint64_t, std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(sparse_matrix_gate,            std::uint64_t, std::uint64_t, std::uint64_t, const SparseMatrix<Prec, Space>&);

#undef INSTANTIATE_FLAT_STATE_OVERLOADS
// clang-format on

}  // namespace scaluq::internal
