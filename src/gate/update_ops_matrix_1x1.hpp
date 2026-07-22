#pragma once

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

}  // namespace scaluq::internal
