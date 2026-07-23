#pragma once

namespace scaluq::internal {

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33,
          typename SimdType,
          UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate_simd(std::uint64_t target_mask,
                                       std::uint64_t control_mask,
                                       std::uint64_t control_value_mask,
                                       const Matrix4x4<State::prec>& matrix,
                                       State& state) {
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t lower_target_mask = -target_mask & target_mask;
    const std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t skip_mask = target_mask | control_mask;
    const auto m00 = SimdCoefFor<SimdType, M00>::splat(matrix[0][0]);
    const auto m01 = SimdCoefFor<SimdType, M01>::splat(matrix[0][1]);
    const auto m02 = SimdCoefFor<SimdType, M02>::splat(matrix[0][2]);
    const auto m03 = SimdCoefFor<SimdType, M03>::splat(matrix[0][3]);
    const auto m10 = SimdCoefFor<SimdType, M10>::splat(matrix[1][0]);
    const auto m11 = SimdCoefFor<SimdType, M11>::splat(matrix[1][1]);
    const auto m12 = SimdCoefFor<SimdType, M12>::splat(matrix[1][2]);
    const auto m13 = SimdCoefFor<SimdType, M13>::splat(matrix[1][3]);
    const auto m20 = SimdCoefFor<SimdType, M20>::splat(matrix[2][0]);
    const auto m21 = SimdCoefFor<SimdType, M21>::splat(matrix[2][1]);
    const auto m22 = SimdCoefFor<SimdType, M22>::splat(matrix[2][2]);
    const auto m23 = SimdCoefFor<SimdType, M23>::splat(matrix[2][3]);
    const auto m30 = SimdCoefFor<SimdType, M30>::splat(matrix[3][0]);
    const auto m31 = SimdCoefFor<SimdType, M31>::splat(matrix[3][1]);
    const auto m32 = SimdCoefFor<SimdType, M32>::splat(matrix[3][2]);
    const auto m33 = SimdCoefFor<SimdType, M33>::splat(matrix[3][3]);

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

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33,
          std::size_t TargetBit,
          UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate_simd_middle(std::uint64_t target_mask,
                                              std::uint64_t control_mask,
                                              std::uint64_t control_value_mask,
                                              const Matrix4x4<State::prec>& matrix,
                                              State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert((1ULL << TargetBit) < complex_lanes);
    const std::uint64_t outer_target_mask = target_mask ^ (1ULL << TargetBit);
    const std::uint64_t skip_mask = outer_target_mask | control_mask;

    constexpr CoefKind K00 = common_coef_kind(M00, M11);
    constexpr CoefKind K01 = common_coef_kind(M01, M10);
    constexpr CoefKind K02 = common_coef_kind(M02, M13);
    constexpr CoefKind K03 = common_coef_kind(M03, M12);
    constexpr CoefKind K20 = common_coef_kind(M20, M31);
    constexpr CoefKind K21 = common_coef_kind(M21, M30);
    constexpr CoefKind K22 = common_coef_kind(M22, M33);
    constexpr CoefKind K23 = common_coef_kind(M23, M32);
    using C00 = SimdCoefFor<SimdType, K00>;
    using C01 = SimdCoefFor<SimdType, K01>;
    using C02 = SimdCoefFor<SimdType, K02>;
    using C03 = SimdCoefFor<SimdType, K03>;
    using C20 = SimdCoefFor<SimdType, K20>;
    using C21 = SimdCoefFor<SimdType, K21>;
    using C22 = SimdCoefFor<SimdType, K22>;
    using C23 = SimdCoefFor<SimdType, K23>;
    const auto m00 = C00::template select_complex_lane_bit<TargetBit>(matrix[0][0], matrix[1][1]);
    const auto m01 = C01::template select_complex_lane_bit<TargetBit>(matrix[0][1], matrix[1][0]);
    const auto m02 = C02::template select_complex_lane_bit<TargetBit>(matrix[0][2], matrix[1][3]);
    const auto m03 = C03::template select_complex_lane_bit<TargetBit>(matrix[0][3], matrix[1][2]);
    const auto m20 = C20::template select_complex_lane_bit<TargetBit>(matrix[2][0], matrix[3][1]);
    const auto m21 = C21::template select_complex_lane_bit<TargetBit>(matrix[2][1], matrix[3][0]);
    const auto m22 = C22::template select_complex_lane_bit<TargetBit>(matrix[2][2], matrix[3][3]);
    const auto m23 = C23::template select_complex_lane_bit<TargetBit>(matrix[2][3], matrix[3][2]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate_simd_middle",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis2 = basis0 | outer_target_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit)>();
            const auto v3 = v2.template permute_complex_lanes_xor<(1ULL << TargetBit)>();
            (m00 * v0 + m01 * v1 + m02 * v2 + m03 * v3).store_aligned(&state.at_unsafe(basis0));
            (m20 * v0 + m21 * v1 + m22 * v2 + m23 * v3).store_aligned(&state.at_unsafe(basis2));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33,
          std::size_t TargetBit0, std::size_t TargetBit1, UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate_simd_low(std::uint64_t control_mask,
                                           std::uint64_t control_value_mask,
                                           const Matrix4x4<State::prec>& matrix,
                                           State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert(TargetBit0 < TargetBit1);
    static_assert((1ULL << TargetBit1) < complex_lanes);

    constexpr CoefKind K0 = common_coef_kind(M00, M11, M22, M33);
    constexpr CoefKind K1 = common_coef_kind(M01, M10, M23, M32);
    constexpr CoefKind K2 = common_coef_kind(M02, M13, M20, M31);
    constexpr CoefKind K3 = common_coef_kind(M03, M12, M21, M30);
    using C0 = SimdCoefFor<SimdType, K0>;
    using C1 = SimdCoefFor<SimdType, K1>;
    using C2 = SimdCoefFor<SimdType, K2>;
    using C3 = SimdCoefFor<SimdType, K3>;
    const auto m0 = C0::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3]);
    const auto m1 = C1::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][1], matrix[1][0], matrix[2][3], matrix[3][2]);
    const auto m2 = C2::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][2], matrix[1][3], matrix[2][0], matrix[3][1]);
    const auto m3 = C3::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][3], matrix[1][2], matrix[2][1], matrix[3][0]);

    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(control_mask);
    Kokkos::parallel_for(
        "two_target_dense_matrix_gate_simd_low",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, control_mask) | control_value_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v2 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit1)>();
            const auto v3 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit1) |
                                                                  (1ULL << TargetBit0)>();
            (m0 * v0 + m1 * v1 + m2 * v2 + m3 * v3).store_aligned(&state.at_unsafe(basis));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33,
          UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                         std::uint64_t control_mask,
                                         std::uint64_t control_value_mask,
                                         const Matrix4x4<State::prec>& matrix,
                                         State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    const auto m00 = ScalarCoef<State::prec, M00>::splat(matrix[0][0]);
    const auto m01 = ScalarCoef<State::prec, M01>::splat(matrix[0][1]);
    const auto m02 = ScalarCoef<State::prec, M02>::splat(matrix[0][2]);
    const auto m03 = ScalarCoef<State::prec, M03>::splat(matrix[0][3]);
    const auto m10 = ScalarCoef<State::prec, M10>::splat(matrix[1][0]);
    const auto m11 = ScalarCoef<State::prec, M11>::splat(matrix[1][1]);
    const auto m12 = ScalarCoef<State::prec, M12>::splat(matrix[1][2]);
    const auto m13 = ScalarCoef<State::prec, M13>::splat(matrix[1][3]);
    const auto m20 = ScalarCoef<State::prec, M20>::splat(matrix[2][0]);
    const auto m21 = ScalarCoef<State::prec, M21>::splat(matrix[2][1]);
    const auto m22 = ScalarCoef<State::prec, M22>::splat(matrix[2][2]);
    const auto m23 = ScalarCoef<State::prec, M23>::splat(matrix[2][3]);
    const auto m30 = ScalarCoef<State::prec, M30>::splat(matrix[3][0]);
    const auto m31 = ScalarCoef<State::prec, M31>::splat(matrix[3][1]);
    const auto m32 = ScalarCoef<State::prec, M32>::splat(matrix[3][2]);
    const auto m33 = ScalarCoef<State::prec, M33>::splat(matrix[3][3]);
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
            const ComplexType v0 = state.at_unsafe(basis_0);
            const ComplexType v1 = state.at_unsafe(basis_1);
            const ComplexType v2 = state.at_unsafe(basis_2);
            const ComplexType v3 = state.at_unsafe(basis_3);
            const auto res0 = m00 * v0 + m01 * v1 + m02 * v2 + m03 * v3;
            const auto res1 = m10 * v0 + m11 * v1 + m12 * v2 + m13 * v3;
            const auto res2 = m20 * v0 + m21 * v1 + m22 * v2 + m23 * v3;
            const auto res3 = m30 * v0 + m31 * v1 + m32 * v2 + m33 * v3;
            state.at_unsafe(basis_0) = static_cast<ComplexType>(res0);
            state.at_unsafe(basis_1) = static_cast<ComplexType>(res1);
            state.at_unsafe(basis_2) = static_cast<ComplexType>(res2);
            state.at_unsafe(basis_3) = static_cast<ComplexType>(res3);
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33,
          UpdatableStateVector State>
// clang-format on
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<State::prec>& matrix,
                                  State& state) {
    if constexpr (supports_gate_simd<State>) {
        constexpr std::size_t complex_lanes = SimdComplex<State::prec>::complex_lanes;
        constexpr std::uint64_t inlane_mask = complex_lanes - 1;
        const std::uint64_t inlane_targets = target_mask & inlane_mask;
        if (can_use_gate_simd(control_mask, state)) {
            if (inlane_targets == 0) {
                // clang-format off
                two_target_dense_matrix_gate_simd<
                    M00, M01, M02, M03,
                    M10, M11, M12, M13,
                    M20, M21, M22, M23,
                    M30, M31, M32, M33,
                    SimdComplex<State::prec>>(
                    target_mask, control_mask, control_value_mask, matrix, state);
                // clang-format on
                return;
            }
            if (inlane_targets == 0b1) {
                // clang-format off
                two_target_dense_matrix_gate_simd_middle<
                    M00, M01, M02, M03,
                    M10, M11, M12, M13,
                    M20, M21, M22, M23,
                    M30, M31, M32, M33,
                    0>(
                    target_mask, control_mask, control_value_mask, matrix, state);
                // clang-format on
                return;
            }
            if constexpr (complex_lanes > 2) {
                if (inlane_targets == 0b10) {
                    // clang-format off
                    two_target_dense_matrix_gate_simd_middle<
                        M00, M01, M02, M03,
                        M10, M11, M12, M13,
                        M20, M21, M22, M23,
                        M30, M31, M32, M33,
                        1>(
                        target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
                // clang-format off
                two_target_dense_matrix_gate_simd_low<
                    M00, M01, M02, M03,
                    M10, M11, M12, M13,
                    M20, M21, M22, M23,
                    M30, M31, M32, M33,
                    0, 1>(
                    control_mask, control_value_mask, matrix, state);
                // clang-format on
                return;
            }
        }
    }
    // clang-format off
    two_target_dense_matrix_gate_scalar<
        M00, M01, M02, M03,
        M10, M11, M12, M13,
        M20, M21, M22, M23,
        M30, M31, M32, M33>(
        target_mask, control_mask, control_value_mask, matrix, state);
    // clang-format on
}

}  // namespace scaluq::internal
