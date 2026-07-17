#include <cstdint>
#include <utility>

#include "update_ops.hpp"

namespace scaluq::internal {

template <CoefKind Kind, UpdatableStateVector State>
void zero_target_dense_matrix_gate_simd(std::uint64_t control_mask,
                                        std::uint64_t control_value_mask,
                                        Complex<State::prec> matrix,
                                        State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    using Coef = SimdCoef<State::prec, Kind>;
    const auto coef = Coef::splat(matrix);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(control_mask);
    Kokkos::parallel_for(
        "zero_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, control_mask) | control_value_mask;
            const auto value = SimdType::load_aligned(&state.at_unsafe(basis));
            (coef * value).store_aligned(&state.at_unsafe(basis));
        });
}

template <CoefKind Kind, UpdatableStateVector State>
void zero_target_dense_matrix_gate_scalar(std::uint64_t control_mask,
                                          std::uint64_t control_value_mask,
                                          Complex<State::prec> matrix,
                                          State& state) {
    using Coef = ScalarCoef<State::prec, Kind>;
    const auto coef = Coef::splat(matrix);
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "zero_target_dense_matrix_gate",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
            state.at_unsafe(basis) =
                static_cast<Complex<State::prec>>(coef * state.at_unsafe(basis));
        });
}

template <CoefKind Kind, UpdatableStateVector State>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<State::prec> matrix,
                                   State& state) {
    if constexpr (supports_gate_simd<State>) {
        if (can_use_gate_simd(control_mask, state)) {
            zero_target_dense_matrix_gate_simd<Kind>(
                control_mask, control_value_mask, matrix, state);
            return;
        }
    }
    zero_target_dense_matrix_gate_scalar<Kind>(control_mask, control_value_mask, matrix, state);
}

template <CoefKind M00, CoefKind M01, CoefKind M10, CoefKind M11, UpdatableStateVector State>
void one_target_dense_matrix_gate_simd(std::uint64_t target_mask,
                                       std::uint64_t control_mask,
                                       std::uint64_t control_value_mask,
                                       const Matrix2x2<State::prec>& matrix,
                                       State& state) {
    using SimdType = SimdComplex<State::prec>;
    using C00 = SimdCoef<State::prec, M00>;
    using C01 = SimdCoef<State::prec, M01>;
    using C10 = SimdCoef<State::prec, M10>;
    using C11 = SimdCoef<State::prec, M11>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const C00 m00 = C00::splat(matrix[0][0]);
    const C01 m01 = C01::splat(matrix[0][1]);
    const C10 m10 = C10::splat(matrix[1][0]);
    const C11 m11 = C11::splat(matrix[1][1]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target_mask;
            const auto value0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto value1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            (m00 * value0 + m01 * value1).store_aligned(&state.at_unsafe(basis0));
            (m10 * value0 + m11 * value1).store_aligned(&state.at_unsafe(basis1));
        });
}

template <typename DiagCoef,
          typename OffDiagCoef,
          std::size_t TargetBit,
          UpdatableStateVector State>
void one_target_dense_matrix_gate_simd_inlane(std::uint64_t control_mask,
                                              std::uint64_t control_value_mask,
                                              const Matrix2x2<State::prec>& matrix,
                                              State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert((1ULL << TargetBit) < complex_lanes);
    const DiagCoef diagonal =
        DiagCoef::template select_complex_lane_bit<TargetBit>(matrix[0][0], matrix[1][1]);
    const OffDiagCoef off_diagonal =
        OffDiagCoef::template select_complex_lane_bit<TargetBit>(matrix[0][1], matrix[1][0]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(control_mask);
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_simd_inlane",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, control_mask) | control_value_mask;
            const auto value = SimdType::load_aligned(&state.at_unsafe(basis));
            const auto paired = value.template permute_complex_lanes_xor<TargetBit>();
            (diagonal * value + off_diagonal * paired).store_aligned(&state.at_unsafe(basis));
        });
}

template <CoefKind M00, CoefKind M01, CoefKind M10, CoefKind M11, UpdatableStateVector State>
void one_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                         std::uint64_t control_mask,
                                         std::uint64_t control_value_mask,
                                         const Matrix2x2<State::prec>& matrix,
                                         State& state) {
    using ComplexType = Complex<State::prec>;
    using S00 = ScalarCoef<State::prec, M00>;
    using S01 = ScalarCoef<State::prec, M01>;
    using S10 = ScalarCoef<State::prec, M10>;
    using S11 = ScalarCoef<State::prec, M11>;
    const S00 m00 = S00::splat(matrix[0][0]);
    const S01 m01 = S01::splat(matrix[0][1]);
    const S10 m10 = S10::splat(matrix[1][0]);
    const S11 m11 = S11::splat(matrix[1][1]);

    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_scalar",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target_mask;
            const ComplexType value0 = state.at_unsafe(basis0);
            const ComplexType value1 = state.at_unsafe(basis1);
            const auto result0 = m00 * value0 + m01 * value1;
            const auto result1 = m10 * value0 + m11 * value1;
            state.at_unsafe(basis0) = static_cast<ComplexType>(result0);
            state.at_unsafe(basis1) = static_cast<ComplexType>(result1);
        });
}

template <CoefKind M00, CoefKind M01, CoefKind M10, CoefKind M11, UpdatableStateVector State>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<State::prec>& matrix,
                                  State& state) {
    if constexpr (supports_gate_simd<State>) {
        using DiagCoef = SimdCoef<State::prec, common_coef_kind(M00, M11)>;
        using OffDiagCoef = SimdCoef<State::prec, common_coef_kind(M01, M10)>;
        if (can_use_gate_simd(target_mask | control_mask, state)) {
            one_target_dense_matrix_gate_simd<M00, M01, M10, M11>(
                target_mask, control_mask, control_value_mask, matrix, state);
            return;
        }
        constexpr std::size_t complex_lanes = SimdComplex<State::prec>::complex_lanes;
        constexpr std::uint64_t inlane_mask = complex_lanes - 1;
        const std::uint64_t span = state.flat_dim() >> std::popcount(control_mask);
        if ((target_mask & inlane_mask) != 0 && (control_mask & inlane_mask) == 0 &&
            span >= complex_lanes) {
            if (target_mask == 0b1) {
                one_target_dense_matrix_gate_simd_inlane<DiagCoef, OffDiagCoef, 0>(
                    control_mask, control_value_mask, matrix, state);
                return;
            }
            if constexpr (complex_lanes > 2) {
                if (target_mask == 0b10) {
                    one_target_dense_matrix_gate_simd_inlane<DiagCoef, OffDiagCoef, 1>(
                        control_mask, control_value_mask, matrix, state);
                    return;
                }
            }
        }
    }
    one_target_dense_matrix_gate_scalar<M00, M01, M10, M11>(
        target_mask, control_mask, control_value_mask, matrix, state);
}

template <UpdatableStateVector State>
void two_target_dense_matrix_gate_simd(std::uint64_t target_mask,
                                       std::uint64_t control_mask,
                                       std::uint64_t control_value_mask,
                                       const Matrix4x4<State::prec>& matrix,
                                       State& state) {
    using SimdType = SimdComplex<State::prec>;
    using Coef = typename SimdType::Coef;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t lower_target_mask = -target_mask & target_mask;
    const std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const Coef m00 = Coef::splat(matrix[0][0]);
    const Coef m01 = Coef::splat(matrix[0][1]);
    const Coef m02 = Coef::splat(matrix[0][2]);
    const Coef m03 = Coef::splat(matrix[0][3]);
    const Coef m10 = Coef::splat(matrix[1][0]);
    const Coef m11 = Coef::splat(matrix[1][1]);
    const Coef m12 = Coef::splat(matrix[1][2]);
    const Coef m13 = Coef::splat(matrix[1][3]);
    const Coef m20 = Coef::splat(matrix[2][0]);
    const Coef m21 = Coef::splat(matrix[2][1]);
    const Coef m22 = Coef::splat(matrix[2][2]);
    const Coef m23 = Coef::splat(matrix[2][3]);
    const Coef m30 = Coef::splat(matrix[3][0]);
    const Coef m31 = Coef::splat(matrix[3][1]);
    const Coef m32 = Coef::splat(matrix[3][2]);
    const Coef m33 = Coef::splat(matrix[3][3]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | lower_target_mask;
            const std::uint64_t basis2 = basis0 | upper_target_mask;
            const std::uint64_t basis3 = basis1 | upper_target_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            const auto v3 = SimdType::load_aligned(&state.at_unsafe(basis3));
            (m00 * v0 + m01 * v1 + m02 * v2 + m03 * v3).store_aligned(&state.at_unsafe(basis0));
            (m10 * v0 + m11 * v1 + m12 * v2 + m13 * v3).store_aligned(&state.at_unsafe(basis1));
            (m20 * v0 + m21 * v1 + m22 * v2 + m23 * v3).store_aligned(&state.at_unsafe(basis2));
            (m30 * v0 + m31 * v1 + m32 * v2 + m33 * v3).store_aligned(&state.at_unsafe(basis3));
        });
}

template <UpdatableStateVector State>
void two_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
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
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<State::prec>& matrix,
                                  State& state) {
    if constexpr (supports_gate_simd<State>) {
        if (can_use_gate_simd(target_mask | control_mask, state)) {
            two_target_dense_matrix_gate_simd(
                target_mask, control_mask, control_value_mask, matrix, state);
            return;
        }
    }
    two_target_dense_matrix_gate_scalar(
        target_mask, control_mask, control_value_mask, matrix, state);
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

INSTANTIATE_FLAT_STATE_OVERLOADS(two_target_dense_matrix_gate,  std::uint64_t, std::uint64_t, std::uint64_t, const Matrix4x4<Prec>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(multi_dense_matrix_gate,       std::uint64_t, std::uint64_t, std::uint64_t, const Matrix<Prec, Space>&);
INSTANTIATE_FLAT_STATE_OVERLOADS(sparse_matrix_gate,            std::uint64_t, std::uint64_t, std::uint64_t, const SparseMatrix<Prec, Space>&);

#define INSTANTIATE_ZERO_TARGET_DENSE(Kind, StateType)                                \
    template void zero_target_dense_matrix_gate<CoefKind::Kind, StateType<Prec, Space>>( \
        std::uint64_t, std::uint64_t, Complex<Prec>, StateType<Prec, Space>&)

#define INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(Kind)               \
    INSTANTIATE_ZERO_TARGET_DENSE(Kind, StateVector);                 \
    INSTANTIATE_ZERO_TARGET_DENSE(Kind, StateVectorBatched)

INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(General);
INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(Real);
INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(Imag);
INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(Zero);
INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES(One);

#undef INSTANTIATE_ZERO_TARGET_DENSE_FOR_STATES
#undef INSTANTIATE_ZERO_TARGET_DENSE

#define INSTANTIATE_TAGGED_DENSE(M00, M01, M10, M11, StateType)                         \
    template void one_target_dense_matrix_gate<                                         \
        CoefKind::M00, CoefKind::M01, CoefKind::M10, CoefKind::M11,                     \
        StateType<Prec, Space>>(std::uint64_t, std::uint64_t, std::uint64_t,             \
                                const Matrix2x2<Prec>&, StateType<Prec, Space>&)

#define INSTANTIATE_TAGGED_DENSE_FOR_STATES(M00, M01, M10, M11)                         \
    INSTANTIATE_TAGGED_DENSE(M00, M01, M10, M11, StateVector);                           \
    INSTANTIATE_TAGGED_DENSE(M00, M01, M10, M11, StateVectorBatched)

INSTANTIATE_TAGGED_DENSE_FOR_STATES(General, General, General, General);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(General, Zero, Zero, General);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(Real, Real, Real, Real);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(Real, Imag, Imag, Real);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(Zero, Imag, Imag, Zero);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(One, Zero, Zero, Zero);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(Zero, Zero, Zero, One);
INSTANTIATE_TAGGED_DENSE_FOR_STATES(Zero, One, One, Zero);

#undef INSTANTIATE_TAGGED_DENSE_FOR_STATES
#undef INSTANTIATE_TAGGED_DENSE

#undef INSTANTIATE_FLAT_STATE_OVERLOADS

// clang-format on

}  // namespace scaluq::internal
