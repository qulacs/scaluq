#pragma once

namespace scaluq::internal {

template <CoefKind M00,
          CoefKind M01,
          CoefKind M10,
          CoefKind M11,
          typename SimdType,
          UpdatableStateVector State>
void one_target_dense_matrix_gate_simd_high(std::uint64_t target_mask,
                                            std::uint64_t control_mask,
                                            std::uint64_t control_value_mask,
                                            const Matrix2x2<State::prec>& matrix,
                                            State& state) {
    using C00 = SimdCoefFor<SimdType, M00>;
    using C01 = SimdCoefFor<SimdType, M01>;
    using C10 = SimdCoefFor<SimdType, M10>;
    using C11 = SimdCoefFor<SimdType, M11>;

    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const C00 m00 = C00::splat(matrix[0][0]);
    const C01 m01 = C01::splat(matrix[0][1]);
    const C10 m10 = C10::splat(matrix[1][0]);
    const C11 m11 = C11::splat(matrix[1][1]);
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_simd_high",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            m01.fma(v1, m00 * v0).store_aligned(&state.at_unsafe(basis0));
            m11.fma(v1, m10 * v0).store_aligned(&state.at_unsafe(basis1));
        });
}

template <CoefKind M00,
          CoefKind M01,
          CoefKind M10,
          CoefKind M11,
          std::size_t TargetBit,
          UpdatableStateVector State>
void one_target_dense_matrix_gate_simd_low(std::uint64_t control_mask,
                                           std::uint64_t control_value_mask,
                                           const Matrix2x2<State::prec>& matrix,
                                           State& state) {
    using SimdType = SimdComplex<State::prec>;
    using C00 = SimdCoefFor<SimdType, common_coef_kind(M00, M11)>;
    using C01 = SimdCoefFor<SimdType, common_coef_kind(M01, M10)>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert((1ULL << TargetBit) < complex_lanes);

    const auto m00 = C00::template select_complex_lane_bit<TargetBit>(matrix[0][0], matrix[1][1]);
    const auto m01 = C01::template select_complex_lane_bit<TargetBit>(matrix[0][1], matrix[1][0]);
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(control_mask);
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_simd_low",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, control_mask) | control_value_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit)>();
            m01.fma(v1, m00 * v0).store_aligned(&state.at_unsafe(basis));
        });
}

template <CoefKind M00, CoefKind M01, CoefKind M10, CoefKind M11, UpdatableStateVector State>
void one_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                         std::uint64_t control_mask,
                                         std::uint64_t control_value_mask,
                                         const Matrix2x2<State::prec>& matrix,
                                         State& state) {
    using ComplexType = Complex<State::prec>;

    const auto m00 = ScalarCoef<State::prec, M00>::splat(matrix[0][0]);
    const auto m01 = ScalarCoef<State::prec, M01>::splat(matrix[0][1]);
    const auto m10 = ScalarCoef<State::prec, M10>::splat(matrix[1][0]);
    const auto m11 = ScalarCoef<State::prec, M11>::splat(matrix[1][1]);
    Kokkos::parallel_for(
        "one_target_dense_matrix_gate_scalar",
        Kokkos::RangePolicy<SpaceType<State::space>>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target_mask;
            const ComplexType v0 = state.at_unsafe(basis0);
            const ComplexType v1 = state.at_unsafe(basis1);
            const auto result0 = m00 * v0 + m01 * v1;
            const auto result1 = m10 * v0 + m11 * v1;
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
        constexpr std::size_t complex_lanes = SimdComplex<State::prec>::complex_lanes;
        constexpr std::uint64_t inlane_mask = complex_lanes - 1;
        const std::uint64_t inlane_target = target_mask & inlane_mask;
        if (can_use_gate_simd(control_mask, state)) {
            if (inlane_target == 0) {
                one_target_dense_matrix_gate_simd_high<M00,
                                                       M01,
                                                       M10,
                                                       M11,
                                                       SimdComplex<State::prec>>(
                    target_mask, control_mask, control_value_mask, matrix, state);
                return;
            }
            if (inlane_target == 0b1) {
                one_target_dense_matrix_gate_simd_low<M00, M01, M10, M11, 0>(
                    control_mask, control_value_mask, matrix, state);
                return;
            }
            if constexpr (complex_lanes > 2) {
                if (inlane_target == 0b10) {
                    one_target_dense_matrix_gate_simd_low<M00, M01, M10, M11, 1>(
                        control_mask, control_value_mask, matrix, state);
                    return;
                }
            }
        }
    }
    one_target_dense_matrix_gate_scalar<M00, M01, M10, M11>(
        target_mask, control_mask, control_value_mask, matrix, state);
}

}  // namespace scaluq::internal
