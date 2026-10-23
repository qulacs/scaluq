#pragma once

namespace scaluq::internal {

// SIMD paths handle zero, one, two, or three target bits within a register.
// Lane-local columns are XOR permutations; their coefficients select the output
// row from the corresponding target bits of each complex lane.
// Prepare coefficients before parallel_for to reuse them across state blocks.

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          typename SimdType, UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate_simd(std::uint64_t target_mask,
                                         std::uint64_t control_mask,
                                         std::uint64_t control_value_mask,
                                         const Matrix8x8<State::prec>& matrix,
                                         State& state) {
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t outer_target_mask = target_mask;
    const std::uint64_t target0 = -target_mask & target_mask;
    const std::uint64_t target1 = -(outer_target_mask ^ target0) & (outer_target_mask ^ target0);
    const std::uint64_t target2 = outer_target_mask ^ target0 ^ target1;
    const std::uint64_t skip_mask = outer_target_mask | control_mask;

    const auto m00 = SimdCoefFor<SimdType, M00>::splat(matrix[0][0]);
    const auto m01 = SimdCoefFor<SimdType, M01>::splat(matrix[0][1]);
    const auto m02 = SimdCoefFor<SimdType, M02>::splat(matrix[0][2]);
    const auto m03 = SimdCoefFor<SimdType, M03>::splat(matrix[0][3]);
    const auto m04 = SimdCoefFor<SimdType, M04>::splat(matrix[0][4]);
    const auto m05 = SimdCoefFor<SimdType, M05>::splat(matrix[0][5]);
    const auto m06 = SimdCoefFor<SimdType, M06>::splat(matrix[0][6]);
    const auto m07 = SimdCoefFor<SimdType, M07>::splat(matrix[0][7]);
    const auto m10 = SimdCoefFor<SimdType, M10>::splat(matrix[1][0]);
    const auto m11 = SimdCoefFor<SimdType, M11>::splat(matrix[1][1]);
    const auto m12 = SimdCoefFor<SimdType, M12>::splat(matrix[1][2]);
    const auto m13 = SimdCoefFor<SimdType, M13>::splat(matrix[1][3]);
    const auto m14 = SimdCoefFor<SimdType, M14>::splat(matrix[1][4]);
    const auto m15 = SimdCoefFor<SimdType, M15>::splat(matrix[1][5]);
    const auto m16 = SimdCoefFor<SimdType, M16>::splat(matrix[1][6]);
    const auto m17 = SimdCoefFor<SimdType, M17>::splat(matrix[1][7]);
    const auto m20 = SimdCoefFor<SimdType, M20>::splat(matrix[2][0]);
    const auto m21 = SimdCoefFor<SimdType, M21>::splat(matrix[2][1]);
    const auto m22 = SimdCoefFor<SimdType, M22>::splat(matrix[2][2]);
    const auto m23 = SimdCoefFor<SimdType, M23>::splat(matrix[2][3]);
    const auto m24 = SimdCoefFor<SimdType, M24>::splat(matrix[2][4]);
    const auto m25 = SimdCoefFor<SimdType, M25>::splat(matrix[2][5]);
    const auto m26 = SimdCoefFor<SimdType, M26>::splat(matrix[2][6]);
    const auto m27 = SimdCoefFor<SimdType, M27>::splat(matrix[2][7]);
    const auto m30 = SimdCoefFor<SimdType, M30>::splat(matrix[3][0]);
    const auto m31 = SimdCoefFor<SimdType, M31>::splat(matrix[3][1]);
    const auto m32 = SimdCoefFor<SimdType, M32>::splat(matrix[3][2]);
    const auto m33 = SimdCoefFor<SimdType, M33>::splat(matrix[3][3]);
    const auto m34 = SimdCoefFor<SimdType, M34>::splat(matrix[3][4]);
    const auto m35 = SimdCoefFor<SimdType, M35>::splat(matrix[3][5]);
    const auto m36 = SimdCoefFor<SimdType, M36>::splat(matrix[3][6]);
    const auto m37 = SimdCoefFor<SimdType, M37>::splat(matrix[3][7]);
    const auto m40 = SimdCoefFor<SimdType, M40>::splat(matrix[4][0]);
    const auto m41 = SimdCoefFor<SimdType, M41>::splat(matrix[4][1]);
    const auto m42 = SimdCoefFor<SimdType, M42>::splat(matrix[4][2]);
    const auto m43 = SimdCoefFor<SimdType, M43>::splat(matrix[4][3]);
    const auto m44 = SimdCoefFor<SimdType, M44>::splat(matrix[4][4]);
    const auto m45 = SimdCoefFor<SimdType, M45>::splat(matrix[4][5]);
    const auto m46 = SimdCoefFor<SimdType, M46>::splat(matrix[4][6]);
    const auto m47 = SimdCoefFor<SimdType, M47>::splat(matrix[4][7]);
    const auto m50 = SimdCoefFor<SimdType, M50>::splat(matrix[5][0]);
    const auto m51 = SimdCoefFor<SimdType, M51>::splat(matrix[5][1]);
    const auto m52 = SimdCoefFor<SimdType, M52>::splat(matrix[5][2]);
    const auto m53 = SimdCoefFor<SimdType, M53>::splat(matrix[5][3]);
    const auto m54 = SimdCoefFor<SimdType, M54>::splat(matrix[5][4]);
    const auto m55 = SimdCoefFor<SimdType, M55>::splat(matrix[5][5]);
    const auto m56 = SimdCoefFor<SimdType, M56>::splat(matrix[5][6]);
    const auto m57 = SimdCoefFor<SimdType, M57>::splat(matrix[5][7]);
    const auto m60 = SimdCoefFor<SimdType, M60>::splat(matrix[6][0]);
    const auto m61 = SimdCoefFor<SimdType, M61>::splat(matrix[6][1]);
    const auto m62 = SimdCoefFor<SimdType, M62>::splat(matrix[6][2]);
    const auto m63 = SimdCoefFor<SimdType, M63>::splat(matrix[6][3]);
    const auto m64 = SimdCoefFor<SimdType, M64>::splat(matrix[6][4]);
    const auto m65 = SimdCoefFor<SimdType, M65>::splat(matrix[6][5]);
    const auto m66 = SimdCoefFor<SimdType, M66>::splat(matrix[6][6]);
    const auto m67 = SimdCoefFor<SimdType, M67>::splat(matrix[6][7]);
    const auto m70 = SimdCoefFor<SimdType, M70>::splat(matrix[7][0]);
    const auto m71 = SimdCoefFor<SimdType, M71>::splat(matrix[7][1]);
    const auto m72 = SimdCoefFor<SimdType, M72>::splat(matrix[7][2]);
    const auto m73 = SimdCoefFor<SimdType, M73>::splat(matrix[7][3]);
    const auto m74 = SimdCoefFor<SimdType, M74>::splat(matrix[7][4]);
    const auto m75 = SimdCoefFor<SimdType, M75>::splat(matrix[7][5]);
    const auto m76 = SimdCoefFor<SimdType, M76>::splat(matrix[7][6]);
    const auto m77 = SimdCoefFor<SimdType, M77>::splat(matrix[7][7]);

    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "three_target_dense_matrix_gate_simd",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(g * complex_lanes, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target0;
            const std::uint64_t basis2 = basis0 | target1;
            const std::uint64_t basis3 = basis0 | target0 | target1;
            const std::uint64_t basis4 = basis0 | target2;
            const std::uint64_t basis5 = basis0 | target0 | target2;
            const std::uint64_t basis6 = basis0 | target1 | target2;
            const std::uint64_t basis7 = basis0 | target0 | target1 | target2;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            const auto v3 = SimdType::load_aligned(&state.at_unsafe(basis3));
            const auto v4 = SimdType::load_aligned(&state.at_unsafe(basis4));
            const auto v5 = SimdType::load_aligned(&state.at_unsafe(basis5));
            const auto v6 = SimdType::load_aligned(&state.at_unsafe(basis6));
            const auto v7 = SimdType::load_aligned(&state.at_unsafe(basis7));
            ((m01.fma(v1, m00 * v0) + m03.fma(v3, m02 * v2)) +
             (m05.fma(v5, m04 * v4) + m07.fma(v7, m06 * v6)))
                .store_aligned(&state.at_unsafe(basis0));
            ((m11.fma(v1, m10 * v0) + m13.fma(v3, m12 * v2)) +
             (m15.fma(v5, m14 * v4) + m17.fma(v7, m16 * v6)))
                .store_aligned(&state.at_unsafe(basis1));
            ((m21.fma(v1, m20 * v0) + m23.fma(v3, m22 * v2)) +
             (m25.fma(v5, m24 * v4) + m27.fma(v7, m26 * v6)))
                .store_aligned(&state.at_unsafe(basis2));
            ((m31.fma(v1, m30 * v0) + m33.fma(v3, m32 * v2)) +
             (m35.fma(v5, m34 * v4) + m37.fma(v7, m36 * v6)))
                .store_aligned(&state.at_unsafe(basis3));
            ((m41.fma(v1, m40 * v0) + m43.fma(v3, m42 * v2)) +
             (m45.fma(v5, m44 * v4) + m47.fma(v7, m46 * v6)))
                .store_aligned(&state.at_unsafe(basis4));
            ((m51.fma(v1, m50 * v0) + m53.fma(v3, m52 * v2)) +
             (m55.fma(v5, m54 * v4) + m57.fma(v7, m56 * v6)))
                .store_aligned(&state.at_unsafe(basis5));
            ((m61.fma(v1, m60 * v0) + m63.fma(v3, m62 * v2)) +
             (m65.fma(v5, m64 * v4) + m67.fma(v7, m66 * v6)))
                .store_aligned(&state.at_unsafe(basis6));
            ((m71.fma(v1, m70 * v0) + m73.fma(v3, m72 * v2)) +
             (m75.fma(v5, m74 * v4) + m77.fma(v7, m76 * v6)))
                .store_aligned(&state.at_unsafe(basis7));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          std::size_t TargetBit0, UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate_simd_middle1(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 std::uint64_t control_value_mask,
                                                 const Matrix8x8<State::prec>& matrix,
                                                 State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert((1ULL << TargetBit0) < complex_lanes);
    const std::uint64_t outer_target_mask = target_mask ^ ((1ULL << TargetBit0));
    const std::uint64_t target0 = -outer_target_mask & outer_target_mask;
    const std::uint64_t target1 = outer_target_mask ^ target0;
    const std::uint64_t skip_mask = outer_target_mask | control_mask;

    using C00 = SimdCoefFor<SimdType, common_coef_kind(M00, M11)>;
    const auto m00 = C00::template select_complex_lane_bit<TargetBit0>(matrix[0][0], matrix[1][1]);
    using C01 = SimdCoefFor<SimdType, common_coef_kind(M01, M10)>;
    const auto m01 = C01::template select_complex_lane_bit<TargetBit0>(matrix[0][1], matrix[1][0]);
    using C02 = SimdCoefFor<SimdType, common_coef_kind(M02, M13)>;
    const auto m02 = C02::template select_complex_lane_bit<TargetBit0>(matrix[0][2], matrix[1][3]);
    using C03 = SimdCoefFor<SimdType, common_coef_kind(M03, M12)>;
    const auto m03 = C03::template select_complex_lane_bit<TargetBit0>(matrix[0][3], matrix[1][2]);
    using C04 = SimdCoefFor<SimdType, common_coef_kind(M04, M15)>;
    const auto m04 = C04::template select_complex_lane_bit<TargetBit0>(matrix[0][4], matrix[1][5]);
    using C05 = SimdCoefFor<SimdType, common_coef_kind(M05, M14)>;
    const auto m05 = C05::template select_complex_lane_bit<TargetBit0>(matrix[0][5], matrix[1][4]);
    using C06 = SimdCoefFor<SimdType, common_coef_kind(M06, M17)>;
    const auto m06 = C06::template select_complex_lane_bit<TargetBit0>(matrix[0][6], matrix[1][7]);
    using C07 = SimdCoefFor<SimdType, common_coef_kind(M07, M16)>;
    const auto m07 = C07::template select_complex_lane_bit<TargetBit0>(matrix[0][7], matrix[1][6]);
    using C20 = SimdCoefFor<SimdType, common_coef_kind(M20, M31)>;
    const auto m20 = C20::template select_complex_lane_bit<TargetBit0>(matrix[2][0], matrix[3][1]);
    using C21 = SimdCoefFor<SimdType, common_coef_kind(M21, M30)>;
    const auto m21 = C21::template select_complex_lane_bit<TargetBit0>(matrix[2][1], matrix[3][0]);
    using C22 = SimdCoefFor<SimdType, common_coef_kind(M22, M33)>;
    const auto m22 = C22::template select_complex_lane_bit<TargetBit0>(matrix[2][2], matrix[3][3]);
    using C23 = SimdCoefFor<SimdType, common_coef_kind(M23, M32)>;
    const auto m23 = C23::template select_complex_lane_bit<TargetBit0>(matrix[2][3], matrix[3][2]);
    using C24 = SimdCoefFor<SimdType, common_coef_kind(M24, M35)>;
    const auto m24 = C24::template select_complex_lane_bit<TargetBit0>(matrix[2][4], matrix[3][5]);
    using C25 = SimdCoefFor<SimdType, common_coef_kind(M25, M34)>;
    const auto m25 = C25::template select_complex_lane_bit<TargetBit0>(matrix[2][5], matrix[3][4]);
    using C26 = SimdCoefFor<SimdType, common_coef_kind(M26, M37)>;
    const auto m26 = C26::template select_complex_lane_bit<TargetBit0>(matrix[2][6], matrix[3][7]);
    using C27 = SimdCoefFor<SimdType, common_coef_kind(M27, M36)>;
    const auto m27 = C27::template select_complex_lane_bit<TargetBit0>(matrix[2][7], matrix[3][6]);
    using C40 = SimdCoefFor<SimdType, common_coef_kind(M40, M51)>;
    const auto m40 = C40::template select_complex_lane_bit<TargetBit0>(matrix[4][0], matrix[5][1]);
    using C41 = SimdCoefFor<SimdType, common_coef_kind(M41, M50)>;
    const auto m41 = C41::template select_complex_lane_bit<TargetBit0>(matrix[4][1], matrix[5][0]);
    using C42 = SimdCoefFor<SimdType, common_coef_kind(M42, M53)>;
    const auto m42 = C42::template select_complex_lane_bit<TargetBit0>(matrix[4][2], matrix[5][3]);
    using C43 = SimdCoefFor<SimdType, common_coef_kind(M43, M52)>;
    const auto m43 = C43::template select_complex_lane_bit<TargetBit0>(matrix[4][3], matrix[5][2]);
    using C44 = SimdCoefFor<SimdType, common_coef_kind(M44, M55)>;
    const auto m44 = C44::template select_complex_lane_bit<TargetBit0>(matrix[4][4], matrix[5][5]);
    using C45 = SimdCoefFor<SimdType, common_coef_kind(M45, M54)>;
    const auto m45 = C45::template select_complex_lane_bit<TargetBit0>(matrix[4][5], matrix[5][4]);
    using C46 = SimdCoefFor<SimdType, common_coef_kind(M46, M57)>;
    const auto m46 = C46::template select_complex_lane_bit<TargetBit0>(matrix[4][6], matrix[5][7]);
    using C47 = SimdCoefFor<SimdType, common_coef_kind(M47, M56)>;
    const auto m47 = C47::template select_complex_lane_bit<TargetBit0>(matrix[4][7], matrix[5][6]);
    using C60 = SimdCoefFor<SimdType, common_coef_kind(M60, M71)>;
    const auto m60 = C60::template select_complex_lane_bit<TargetBit0>(matrix[6][0], matrix[7][1]);
    using C61 = SimdCoefFor<SimdType, common_coef_kind(M61, M70)>;
    const auto m61 = C61::template select_complex_lane_bit<TargetBit0>(matrix[6][1], matrix[7][0]);
    using C62 = SimdCoefFor<SimdType, common_coef_kind(M62, M73)>;
    const auto m62 = C62::template select_complex_lane_bit<TargetBit0>(matrix[6][2], matrix[7][3]);
    using C63 = SimdCoefFor<SimdType, common_coef_kind(M63, M72)>;
    const auto m63 = C63::template select_complex_lane_bit<TargetBit0>(matrix[6][3], matrix[7][2]);
    using C64 = SimdCoefFor<SimdType, common_coef_kind(M64, M75)>;
    const auto m64 = C64::template select_complex_lane_bit<TargetBit0>(matrix[6][4], matrix[7][5]);
    using C65 = SimdCoefFor<SimdType, common_coef_kind(M65, M74)>;
    const auto m65 = C65::template select_complex_lane_bit<TargetBit0>(matrix[6][5], matrix[7][4]);
    using C66 = SimdCoefFor<SimdType, common_coef_kind(M66, M77)>;
    const auto m66 = C66::template select_complex_lane_bit<TargetBit0>(matrix[6][6], matrix[7][7]);
    using C67 = SimdCoefFor<SimdType, common_coef_kind(M67, M76)>;
    const auto m67 = C67::template select_complex_lane_bit<TargetBit0>(matrix[6][7], matrix[7][6]);

    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "three_target_dense_matrix_gate_simd_middle1",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(g * complex_lanes, skip_mask) | control_value_mask;
            const std::uint64_t basis2 = basis0 | target0;
            const std::uint64_t basis4 = basis0 | target1;
            const std::uint64_t basis6 = basis0 | target0 | target1;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            const auto v3 = v2.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v4 = SimdType::load_aligned(&state.at_unsafe(basis4));
            const auto v5 = v4.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v6 = SimdType::load_aligned(&state.at_unsafe(basis6));
            const auto v7 = v6.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            ((m01.fma(v1, m00 * v0) + m03.fma(v3, m02 * v2)) +
             (m05.fma(v5, m04 * v4) + m07.fma(v7, m06 * v6)))
                .store_aligned(&state.at_unsafe(basis0));
            ((m21.fma(v1, m20 * v0) + m23.fma(v3, m22 * v2)) +
             (m25.fma(v5, m24 * v4) + m27.fma(v7, m26 * v6)))
                .store_aligned(&state.at_unsafe(basis2));
            ((m41.fma(v1, m40 * v0) + m43.fma(v3, m42 * v2)) +
             (m45.fma(v5, m44 * v4) + m47.fma(v7, m46 * v6)))
                .store_aligned(&state.at_unsafe(basis4));
            ((m61.fma(v1, m60 * v0) + m63.fma(v3, m62 * v2)) +
             (m65.fma(v5, m64 * v4) + m67.fma(v7, m66 * v6)))
                .store_aligned(&state.at_unsafe(basis6));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          std::size_t TargetBit0, std::size_t TargetBit1, UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate_simd_middle2(std::uint64_t target_mask,
                                                 std::uint64_t control_mask,
                                                 std::uint64_t control_value_mask,
                                                 const Matrix8x8<State::prec>& matrix,
                                                 State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert(TargetBit0 < TargetBit1);
    static_assert((1ULL << TargetBit1) < complex_lanes);
    const std::uint64_t outer_target_mask =
        target_mask ^ ((1ULL << TargetBit0) | (1ULL << TargetBit1));
    const std::uint64_t target0 = outer_target_mask;
    const std::uint64_t skip_mask = outer_target_mask | control_mask;

    using C00 = SimdCoefFor<SimdType, common_coef_kind(M00, M11, M22, M33)>;
    const auto m00 = C00::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][3]);
    using C01 = SimdCoefFor<SimdType, common_coef_kind(M01, M10, M23, M32)>;
    const auto m01 = C01::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][1], matrix[1][0], matrix[2][3], matrix[3][2]);
    using C02 = SimdCoefFor<SimdType, common_coef_kind(M02, M13, M20, M31)>;
    const auto m02 = C02::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][2], matrix[1][3], matrix[2][0], matrix[3][1]);
    using C03 = SimdCoefFor<SimdType, common_coef_kind(M03, M12, M21, M30)>;
    const auto m03 = C03::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][3], matrix[1][2], matrix[2][1], matrix[3][0]);
    using C04 = SimdCoefFor<SimdType, common_coef_kind(M04, M15, M26, M37)>;
    const auto m04 = C04::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][4], matrix[1][5], matrix[2][6], matrix[3][7]);
    using C05 = SimdCoefFor<SimdType, common_coef_kind(M05, M14, M27, M36)>;
    const auto m05 = C05::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][5], matrix[1][4], matrix[2][7], matrix[3][6]);
    using C06 = SimdCoefFor<SimdType, common_coef_kind(M06, M17, M24, M35)>;
    const auto m06 = C06::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][6], matrix[1][7], matrix[2][4], matrix[3][5]);
    using C07 = SimdCoefFor<SimdType, common_coef_kind(M07, M16, M25, M34)>;
    const auto m07 = C07::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[0][7], matrix[1][6], matrix[2][5], matrix[3][4]);
    using C40 = SimdCoefFor<SimdType, common_coef_kind(M40, M51, M62, M73)>;
    const auto m40 = C40::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][0], matrix[5][1], matrix[6][2], matrix[7][3]);
    using C41 = SimdCoefFor<SimdType, common_coef_kind(M41, M50, M63, M72)>;
    const auto m41 = C41::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][1], matrix[5][0], matrix[6][3], matrix[7][2]);
    using C42 = SimdCoefFor<SimdType, common_coef_kind(M42, M53, M60, M71)>;
    const auto m42 = C42::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][2], matrix[5][3], matrix[6][0], matrix[7][1]);
    using C43 = SimdCoefFor<SimdType, common_coef_kind(M43, M52, M61, M70)>;
    const auto m43 = C43::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][3], matrix[5][2], matrix[6][1], matrix[7][0]);
    using C44 = SimdCoefFor<SimdType, common_coef_kind(M44, M55, M66, M77)>;
    const auto m44 = C44::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][4], matrix[5][5], matrix[6][6], matrix[7][7]);
    using C45 = SimdCoefFor<SimdType, common_coef_kind(M45, M54, M67, M76)>;
    const auto m45 = C45::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][5], matrix[5][4], matrix[6][7], matrix[7][6]);
    using C46 = SimdCoefFor<SimdType, common_coef_kind(M46, M57, M64, M75)>;
    const auto m46 = C46::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][6], matrix[5][7], matrix[6][4], matrix[7][5]);
    using C47 = SimdCoefFor<SimdType, common_coef_kind(M47, M56, M65, M74)>;
    const auto m47 = C47::template select_complex_lane_bits<TargetBit0, TargetBit1>(
        matrix[4][7], matrix[5][6], matrix[6][5], matrix[7][4]);

    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "three_target_dense_matrix_gate_simd_middle2",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(g * complex_lanes, skip_mask) | control_value_mask;
            const std::uint64_t basis4 = basis0 | target0;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v2 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit1)>();
            const auto v3 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0) |
                                                                  (1ULL << TargetBit1)>();
            const auto v4 = SimdType::load_aligned(&state.at_unsafe(basis4));
            const auto v5 = v4.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v6 = v4.template permute_complex_lanes_xor<(1ULL << TargetBit1)>();
            const auto v7 = v4.template permute_complex_lanes_xor<(1ULL << TargetBit0) |
                                                                  (1ULL << TargetBit1)>();
            ((m01.fma(v1, m00 * v0) + m03.fma(v3, m02 * v2)) +
             (m05.fma(v5, m04 * v4) + m07.fma(v7, m06 * v6)))
                .store_aligned(&state.at_unsafe(basis0));
            ((m41.fma(v1, m40 * v0) + m43.fma(v3, m42 * v2)) +
             (m45.fma(v5, m44 * v4) + m47.fma(v7, m46 * v6)))
                .store_aligned(&state.at_unsafe(basis4));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          std::size_t TargetBit0, std::size_t TargetBit1, std::size_t TargetBit2, UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate_simd_low(std::uint64_t control_mask,
                                             std::uint64_t control_value_mask,
                                             const Matrix8x8<State::prec>& matrix,
                                             State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    static_assert(TargetBit0 < TargetBit1 && TargetBit1 < TargetBit2);
    static_assert((1ULL << TargetBit2) < complex_lanes);
    const std::uint64_t skip_mask = control_mask;

    using C00 = SimdCoefFor<SimdType, common_coef_kind(M00, M11, M22, M33, M44, M55, M66, M77)>;
    const auto m00 =
        C00::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][0],
                                                                                   matrix[1][1],
                                                                                   matrix[2][2],
                                                                                   matrix[3][3],
                                                                                   matrix[4][4],
                                                                                   matrix[5][5],
                                                                                   matrix[6][6],
                                                                                   matrix[7][7]);
    using C01 = SimdCoefFor<SimdType, common_coef_kind(M01, M10, M23, M32, M45, M54, M67, M76)>;
    const auto m01 =
        C01::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][1],
                                                                                   matrix[1][0],
                                                                                   matrix[2][3],
                                                                                   matrix[3][2],
                                                                                   matrix[4][5],
                                                                                   matrix[5][4],
                                                                                   matrix[6][7],
                                                                                   matrix[7][6]);
    using C02 = SimdCoefFor<SimdType, common_coef_kind(M02, M13, M20, M31, M46, M57, M64, M75)>;
    const auto m02 =
        C02::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][2],
                                                                                   matrix[1][3],
                                                                                   matrix[2][0],
                                                                                   matrix[3][1],
                                                                                   matrix[4][6],
                                                                                   matrix[5][7],
                                                                                   matrix[6][4],
                                                                                   matrix[7][5]);
    using C03 = SimdCoefFor<SimdType, common_coef_kind(M03, M12, M21, M30, M47, M56, M65, M74)>;
    const auto m03 =
        C03::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][3],
                                                                                   matrix[1][2],
                                                                                   matrix[2][1],
                                                                                   matrix[3][0],
                                                                                   matrix[4][7],
                                                                                   matrix[5][6],
                                                                                   matrix[6][5],
                                                                                   matrix[7][4]);
    using C04 = SimdCoefFor<SimdType, common_coef_kind(M04, M15, M26, M37, M40, M51, M62, M73)>;
    const auto m04 =
        C04::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][4],
                                                                                   matrix[1][5],
                                                                                   matrix[2][6],
                                                                                   matrix[3][7],
                                                                                   matrix[4][0],
                                                                                   matrix[5][1],
                                                                                   matrix[6][2],
                                                                                   matrix[7][3]);
    using C05 = SimdCoefFor<SimdType, common_coef_kind(M05, M14, M27, M36, M41, M50, M63, M72)>;
    const auto m05 =
        C05::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][5],
                                                                                   matrix[1][4],
                                                                                   matrix[2][7],
                                                                                   matrix[3][6],
                                                                                   matrix[4][1],
                                                                                   matrix[5][0],
                                                                                   matrix[6][3],
                                                                                   matrix[7][2]);
    using C06 = SimdCoefFor<SimdType, common_coef_kind(M06, M17, M24, M35, M42, M53, M60, M71)>;
    const auto m06 =
        C06::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][6],
                                                                                   matrix[1][7],
                                                                                   matrix[2][4],
                                                                                   matrix[3][5],
                                                                                   matrix[4][2],
                                                                                   matrix[5][3],
                                                                                   matrix[6][0],
                                                                                   matrix[7][1]);
    using C07 = SimdCoefFor<SimdType, common_coef_kind(M07, M16, M25, M34, M43, M52, M61, M70)>;
    const auto m07 =
        C07::template select_complex_lane_bits<TargetBit0, TargetBit1, TargetBit2>(matrix[0][7],
                                                                                   matrix[1][6],
                                                                                   matrix[2][5],
                                                                                   matrix[3][4],
                                                                                   matrix[4][3],
                                                                                   matrix[5][2],
                                                                                   matrix[6][1],
                                                                                   matrix[7][0]);

    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "three_target_dense_matrix_gate_simd_low",
        Kokkos::RangePolicy<SpaceType<State::space>>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(g * complex_lanes, skip_mask) | control_value_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0)>();
            const auto v2 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit1)>();
            const auto v3 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0) |
                                                                  (1ULL << TargetBit1)>();
            const auto v4 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit2)>();
            const auto v5 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit0) |
                                                                  (1ULL << TargetBit2)>();
            const auto v6 = v0.template permute_complex_lanes_xor<(1ULL << TargetBit1) |
                                                                  (1ULL << TargetBit2)>();
            const auto v7 =
                v0.template permute_complex_lanes_xor<(1ULL << TargetBit0) | (1ULL << TargetBit1) |
                                                      (1ULL << TargetBit2)>();
            ((m01.fma(v1, m00 * v0) + m03.fma(v3, m02 * v2)) +
             (m05.fma(v5, m04 * v4) + m07.fma(v7, m06 * v6)))
                .store_aligned(&state.at_unsafe(basis0));
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate_scalar(std::uint64_t target_mask,
                                           std::uint64_t control_mask,
                                           std::uint64_t control_value_mask,
                                           const Matrix8x8<State::prec>& matrix,
                                           State& state) {
    using ComplexType = Complex<State::prec>;
    const auto m00 = ScalarCoef<State::prec, M00>::splat(matrix[0][0]);
    const auto m01 = ScalarCoef<State::prec, M01>::splat(matrix[0][1]);
    const auto m02 = ScalarCoef<State::prec, M02>::splat(matrix[0][2]);
    const auto m03 = ScalarCoef<State::prec, M03>::splat(matrix[0][3]);
    const auto m04 = ScalarCoef<State::prec, M04>::splat(matrix[0][4]);
    const auto m05 = ScalarCoef<State::prec, M05>::splat(matrix[0][5]);
    const auto m06 = ScalarCoef<State::prec, M06>::splat(matrix[0][6]);
    const auto m07 = ScalarCoef<State::prec, M07>::splat(matrix[0][7]);
    const auto m10 = ScalarCoef<State::prec, M10>::splat(matrix[1][0]);
    const auto m11 = ScalarCoef<State::prec, M11>::splat(matrix[1][1]);
    const auto m12 = ScalarCoef<State::prec, M12>::splat(matrix[1][2]);
    const auto m13 = ScalarCoef<State::prec, M13>::splat(matrix[1][3]);
    const auto m14 = ScalarCoef<State::prec, M14>::splat(matrix[1][4]);
    const auto m15 = ScalarCoef<State::prec, M15>::splat(matrix[1][5]);
    const auto m16 = ScalarCoef<State::prec, M16>::splat(matrix[1][6]);
    const auto m17 = ScalarCoef<State::prec, M17>::splat(matrix[1][7]);
    const auto m20 = ScalarCoef<State::prec, M20>::splat(matrix[2][0]);
    const auto m21 = ScalarCoef<State::prec, M21>::splat(matrix[2][1]);
    const auto m22 = ScalarCoef<State::prec, M22>::splat(matrix[2][2]);
    const auto m23 = ScalarCoef<State::prec, M23>::splat(matrix[2][3]);
    const auto m24 = ScalarCoef<State::prec, M24>::splat(matrix[2][4]);
    const auto m25 = ScalarCoef<State::prec, M25>::splat(matrix[2][5]);
    const auto m26 = ScalarCoef<State::prec, M26>::splat(matrix[2][6]);
    const auto m27 = ScalarCoef<State::prec, M27>::splat(matrix[2][7]);
    const auto m30 = ScalarCoef<State::prec, M30>::splat(matrix[3][0]);
    const auto m31 = ScalarCoef<State::prec, M31>::splat(matrix[3][1]);
    const auto m32 = ScalarCoef<State::prec, M32>::splat(matrix[3][2]);
    const auto m33 = ScalarCoef<State::prec, M33>::splat(matrix[3][3]);
    const auto m34 = ScalarCoef<State::prec, M34>::splat(matrix[3][4]);
    const auto m35 = ScalarCoef<State::prec, M35>::splat(matrix[3][5]);
    const auto m36 = ScalarCoef<State::prec, M36>::splat(matrix[3][6]);
    const auto m37 = ScalarCoef<State::prec, M37>::splat(matrix[3][7]);
    const auto m40 = ScalarCoef<State::prec, M40>::splat(matrix[4][0]);
    const auto m41 = ScalarCoef<State::prec, M41>::splat(matrix[4][1]);
    const auto m42 = ScalarCoef<State::prec, M42>::splat(matrix[4][2]);
    const auto m43 = ScalarCoef<State::prec, M43>::splat(matrix[4][3]);
    const auto m44 = ScalarCoef<State::prec, M44>::splat(matrix[4][4]);
    const auto m45 = ScalarCoef<State::prec, M45>::splat(matrix[4][5]);
    const auto m46 = ScalarCoef<State::prec, M46>::splat(matrix[4][6]);
    const auto m47 = ScalarCoef<State::prec, M47>::splat(matrix[4][7]);
    const auto m50 = ScalarCoef<State::prec, M50>::splat(matrix[5][0]);
    const auto m51 = ScalarCoef<State::prec, M51>::splat(matrix[5][1]);
    const auto m52 = ScalarCoef<State::prec, M52>::splat(matrix[5][2]);
    const auto m53 = ScalarCoef<State::prec, M53>::splat(matrix[5][3]);
    const auto m54 = ScalarCoef<State::prec, M54>::splat(matrix[5][4]);
    const auto m55 = ScalarCoef<State::prec, M55>::splat(matrix[5][5]);
    const auto m56 = ScalarCoef<State::prec, M56>::splat(matrix[5][6]);
    const auto m57 = ScalarCoef<State::prec, M57>::splat(matrix[5][7]);
    const auto m60 = ScalarCoef<State::prec, M60>::splat(matrix[6][0]);
    const auto m61 = ScalarCoef<State::prec, M61>::splat(matrix[6][1]);
    const auto m62 = ScalarCoef<State::prec, M62>::splat(matrix[6][2]);
    const auto m63 = ScalarCoef<State::prec, M63>::splat(matrix[6][3]);
    const auto m64 = ScalarCoef<State::prec, M64>::splat(matrix[6][4]);
    const auto m65 = ScalarCoef<State::prec, M65>::splat(matrix[6][5]);
    const auto m66 = ScalarCoef<State::prec, M66>::splat(matrix[6][6]);
    const auto m67 = ScalarCoef<State::prec, M67>::splat(matrix[6][7]);
    const auto m70 = ScalarCoef<State::prec, M70>::splat(matrix[7][0]);
    const auto m71 = ScalarCoef<State::prec, M71>::splat(matrix[7][1]);
    const auto m72 = ScalarCoef<State::prec, M72>::splat(matrix[7][2]);
    const auto m73 = ScalarCoef<State::prec, M73>::splat(matrix[7][3]);
    const auto m74 = ScalarCoef<State::prec, M74>::splat(matrix[7][4]);
    const auto m75 = ScalarCoef<State::prec, M75>::splat(matrix[7][5]);
    const auto m76 = ScalarCoef<State::prec, M76>::splat(matrix[7][6]);
    const auto m77 = ScalarCoef<State::prec, M77>::splat(matrix[7][7]);
    const std::uint64_t target0 = -target_mask & target_mask;
    const std::uint64_t remaining = target_mask ^ target0;
    const std::uint64_t target1 = -remaining & remaining;
    const std::uint64_t target2 = remaining ^ target1;
    const std::uint64_t skip_mask = target_mask | control_mask;
    Kokkos::parallel_for(
        "three_target_dense_matrix_gate_scalar",
        Kokkos::RangePolicy<SpaceType<State::space>>(0,
                                                     state.flat_dim() >> std::popcount(skip_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(it, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | target0;
            const std::uint64_t basis2 = basis0 | target1;
            const std::uint64_t basis3 = basis0 | target0 | target1;
            const std::uint64_t basis4 = basis0 | target2;
            const std::uint64_t basis5 = basis0 | target0 | target2;
            const std::uint64_t basis6 = basis0 | target1 | target2;
            const std::uint64_t basis7 = basis0 | target0 | target1 | target2;
            const ComplexType v0 = state.at_unsafe(basis0);
            const ComplexType v1 = state.at_unsafe(basis1);
            const ComplexType v2 = state.at_unsafe(basis2);
            const ComplexType v3 = state.at_unsafe(basis3);
            const ComplexType v4 = state.at_unsafe(basis4);
            const ComplexType v5 = state.at_unsafe(basis5);
            const ComplexType v6 = state.at_unsafe(basis6);
            const ComplexType v7 = state.at_unsafe(basis7);
            state.at_unsafe(basis0) =
                static_cast<ComplexType>(m00 * v0 + m01 * v1 + m02 * v2 + m03 * v3 + m04 * v4 +
                                         m05 * v5 + m06 * v6 + m07 * v7);
            state.at_unsafe(basis1) =
                static_cast<ComplexType>(m10 * v0 + m11 * v1 + m12 * v2 + m13 * v3 + m14 * v4 +
                                         m15 * v5 + m16 * v6 + m17 * v7);
            state.at_unsafe(basis2) =
                static_cast<ComplexType>(m20 * v0 + m21 * v1 + m22 * v2 + m23 * v3 + m24 * v4 +
                                         m25 * v5 + m26 * v6 + m27 * v7);
            state.at_unsafe(basis3) =
                static_cast<ComplexType>(m30 * v0 + m31 * v1 + m32 * v2 + m33 * v3 + m34 * v4 +
                                         m35 * v5 + m36 * v6 + m37 * v7);
            state.at_unsafe(basis4) =
                static_cast<ComplexType>(m40 * v0 + m41 * v1 + m42 * v2 + m43 * v3 + m44 * v4 +
                                         m45 * v5 + m46 * v6 + m47 * v7);
            state.at_unsafe(basis5) =
                static_cast<ComplexType>(m50 * v0 + m51 * v1 + m52 * v2 + m53 * v3 + m54 * v4 +
                                         m55 * v5 + m56 * v6 + m57 * v7);
            state.at_unsafe(basis6) =
                static_cast<ComplexType>(m60 * v0 + m61 * v1 + m62 * v2 + m63 * v3 + m64 * v4 +
                                         m65 * v5 + m66 * v6 + m67 * v7);
            state.at_unsafe(basis7) =
                static_cast<ComplexType>(m70 * v0 + m71 * v1 + m72 * v2 + m73 * v3 + m74 * v4 +
                                         m75 * v5 + m76 * v6 + m77 * v7);
        });
}

// clang-format off
template <CoefKind M00, CoefKind M01, CoefKind M02, CoefKind M03, CoefKind M04, CoefKind M05, CoefKind M06, CoefKind M07,
          CoefKind M10, CoefKind M11, CoefKind M12, CoefKind M13, CoefKind M14, CoefKind M15, CoefKind M16, CoefKind M17,
          CoefKind M20, CoefKind M21, CoefKind M22, CoefKind M23, CoefKind M24, CoefKind M25, CoefKind M26, CoefKind M27,
          CoefKind M30, CoefKind M31, CoefKind M32, CoefKind M33, CoefKind M34, CoefKind M35, CoefKind M36, CoefKind M37,
          CoefKind M40, CoefKind M41, CoefKind M42, CoefKind M43, CoefKind M44, CoefKind M45, CoefKind M46, CoefKind M47,
          CoefKind M50, CoefKind M51, CoefKind M52, CoefKind M53, CoefKind M54, CoefKind M55, CoefKind M56, CoefKind M57,
          CoefKind M60, CoefKind M61, CoefKind M62, CoefKind M63, CoefKind M64, CoefKind M65, CoefKind M66, CoefKind M67,
          CoefKind M70, CoefKind M71, CoefKind M72, CoefKind M73, CoefKind M74, CoefKind M75, CoefKind M76, CoefKind M77,
          UpdatableStateVector State>
// clang-format on
void three_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix8x8<State::prec>& matrix,
                                    State& state) {
    if constexpr (supports_gate_simd<State>) {
        constexpr std::size_t complex_lanes = SimdComplex<State::prec>::complex_lanes;
        const std::uint64_t inlane_targets = target_mask & (complex_lanes - 1);
        if (can_use_gate_simd(control_mask, state)) {
            if (inlane_targets == 0b0) {
                // clang-format off
                three_target_dense_matrix_gate_simd<
                    M00, M01, M02, M03, M04, M05, M06, M07,
                    M10, M11, M12, M13, M14, M15, M16, M17,
                    M20, M21, M22, M23, M24, M25, M26, M27,
                    M30, M31, M32, M33, M34, M35, M36, M37,
                    M40, M41, M42, M43, M44, M45, M46, M47,
                    M50, M51, M52, M53, M54, M55, M56, M57,
                    M60, M61, M62, M63, M64, M65, M66, M67,
                    M70, M71, M72, M73, M74, M75, M76, M77,
                    SimdComplex<State::prec>>(target_mask, control_mask, control_value_mask, matrix, state);
                // clang-format on
                return;
            }
            if (inlane_targets == 0b1) {
                // clang-format off
                three_target_dense_matrix_gate_simd_middle1<
                    M00, M01, M02, M03, M04, M05, M06, M07,
                    M10, M11, M12, M13, M14, M15, M16, M17,
                    M20, M21, M22, M23, M24, M25, M26, M27,
                    M30, M31, M32, M33, M34, M35, M36, M37,
                    M40, M41, M42, M43, M44, M45, M46, M47,
                    M50, M51, M52, M53, M54, M55, M56, M57,
                    M60, M61, M62, M63, M64, M65, M66, M67,
                    M70, M71, M72, M73, M74, M75, M76, M77,
                    0>(target_mask, control_mask, control_value_mask, matrix, state);
                // clang-format on
                return;
            }
            if constexpr (complex_lanes > 2) {
                if (inlane_targets == 0b10) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_middle1<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        1>(target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
                if (inlane_targets == 0b11) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_middle2<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        0, 1>(target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
            }
            if constexpr (complex_lanes > 4) {
                if (inlane_targets == 0b100) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_middle1<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        2>(target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
                if (inlane_targets == 0b101) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_middle2<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        0, 2>(target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
                if (inlane_targets == 0b110) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_middle2<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        1, 2>(target_mask, control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
                if (inlane_targets == 0b111) {
                    // clang-format off
                    three_target_dense_matrix_gate_simd_low<
                        M00, M01, M02, M03, M04, M05, M06, M07,
                        M10, M11, M12, M13, M14, M15, M16, M17,
                        M20, M21, M22, M23, M24, M25, M26, M27,
                        M30, M31, M32, M33, M34, M35, M36, M37,
                        M40, M41, M42, M43, M44, M45, M46, M47,
                        M50, M51, M52, M53, M54, M55, M56, M57,
                        M60, M61, M62, M63, M64, M65, M66, M67,
                        M70, M71, M72, M73, M74, M75, M76, M77,
                        0, 1, 2>(control_mask, control_value_mask, matrix, state);
                    // clang-format on
                    return;
                }
            }
        }
    }
    // clang-format off
    three_target_dense_matrix_gate_scalar<
        M00, M01, M02, M03, M04, M05, M06, M07,
        M10, M11, M12, M13, M14, M15, M16, M17,
        M20, M21, M22, M23, M24, M25, M26, M27,
        M30, M31, M32, M33, M34, M35, M36, M37,
        M40, M41, M42, M43, M44, M45, M46, M47,
        M50, M51, M52, M53, M54, M55, M56, M57,
        M60, M61, M62, M63, M64, M65, M66, M67,
        M70, M71, M72, M73, M74, M75, M76, M77>
        (target_mask, control_mask, control_value_mask, matrix, state);
    // clang-format on
}

}  // namespace scaluq::internal
