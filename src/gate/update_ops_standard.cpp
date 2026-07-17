#include "update_ops.hpp"

namespace scaluq::internal {

template <>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "rx_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params(batch_id) * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            Matrix2x2<Prec> matrix = {
                {{{cosval, Complex<Prec>(0, -sinval)}}, {{Complex<Prec>(0, -sinval), cosval}}}};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Prec> val0 = states.at_unsafe(batch_id, basis_0);
                    Complex<Prec> val1 = states.at_unsafe(batch_id, basis_1);
                    Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states.at_unsafe(batch_id, basis_0) = res0;
                    states.at_unsafe(batch_id, basis_1) = res1;
                });
        });
}

template <>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "ry_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params(batch_id) * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            Matrix2x2<Prec> matrix = {{{{cosval, -sinval}}, {{sinval, cosval}}}};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    Complex<Prec> val0 = states.at_unsafe(batch_id, basis_0);
                    Complex<Prec> val1 = states.at_unsafe(batch_id, basis_1);
                    Complex<Prec> res0 = matrix[0][0] * val0 + matrix[0][1] * val1;
                    Complex<Prec> res1 = matrix[1][0] * val0 + matrix[1][1] * val1;
                    states.at_unsafe(batch_id, basis_0) = res0;
                    states.at_unsafe(batch_id, basis_1) = res1;
                });
        });
}

template <>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states) {
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "rz_gate",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team_member) {
            const std::uint64_t batch_id = team_member.league_rank();
            const Float<Prec> angle = params(batch_id) * pcoef;
            const Float<Prec> cosval = internal::cos(angle / Float<Prec>{2});
            const Float<Prec> sinval = internal::sin(angle / Float<Prec>{2});
            DiagonalMatrix2x2<Prec> diag = {Complex<Prec>(cosval, -sinval),
                                            Complex<Prec>(cosval, sinval)};
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member,
                                        states.dim() >> std::popcount(target_mask | control_mask)),
                [&](std::uint64_t it) {
                    std::uint64_t basis_0 =
                        insert_zero_at_mask_positions(it, control_mask | target_mask) |
                        control_value_mask;
                    std::uint64_t basis_1 = basis_0 | target_mask;
                    states.at_unsafe(batch_id, basis_0) *= diag[0];
                    states.at_unsafe(batch_id, basis_1) *= diag[1];
                });
        });
}

template <UpdatableStateVector State>
void swap_gate_simd(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const std::uint64_t lower_target_mask = target_mask & -target_mask;
    const std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    const std::uint64_t skip_mask = target_mask | control_mask;
    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "swap_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis | lower_target_mask;
            const std::uint64_t basis2 = basis | upper_target_mask;
            const auto v1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            v2.store_aligned(&state.at_unsafe(basis1));
            v1.store_aligned(&state.at_unsafe(basis2));
        });
}

template <UpdatableStateVector State>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               State& state) {
    if constexpr (supports_gate_simd<State>) {
        if (can_use_gate_simd(target_mask | control_mask, state)) {
            swap_gate_simd(target_mask, control_mask, control_value_mask, state);
            return;
        }
    }
    std::uint64_t lower_target_mask = target_mask & -target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "swap_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_value_mask;
            Kokkos::kokkos_swap(state.at_unsafe(basis | lower_target_mask),
                                state.at_unsafe(basis | upper_target_mask));
        });
}

template <UpdatableStateVector State>
void ecr_gate_simd(std::uint64_t physical_target_mask,
                   std::uint64_t physical_control_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    using SimdType = SimdComplex<State::prec>;
    constexpr std::size_t complex_lanes = SimdType::complex_lanes;
    const auto inv_sqrt2 = SimdType::RCoef::splat(Float<State::prec>(INVERSE_SQRT2()));
    const auto plus_i = SimdType::ICoef::splat(Float<State::prec>(1));
    const auto minus_i = SimdType::ICoef::splat(Float<State::prec>(-1));
    const std::uint64_t skip_mask = physical_target_mask | physical_control_mask | control_mask;
    using ExecSpace = SpaceType<State::space>;
    const std::uint64_t flat_span = state.flat_dim() >> std::popcount(skip_mask);
    Kokkos::parallel_for(
        "ecr_gate_simd",
        Kokkos::RangePolicy<ExecSpace>(0, flat_span / complex_lanes),
        KOKKOS_LAMBDA(std::uint64_t g) {
            const std::uint64_t compressed_base = g * complex_lanes;
            const std::uint64_t basis0 =
                insert_zero_at_mask_positions(compressed_base, skip_mask) | control_value_mask;
            const std::uint64_t basis1 = basis0 | physical_control_mask;
            const std::uint64_t basis2 = basis0 | physical_target_mask;
            const std::uint64_t basis3 = basis1 | physical_target_mask;
            const auto v0 = SimdType::load_aligned(&state.at_unsafe(basis0));
            const auto v1 = SimdType::load_aligned(&state.at_unsafe(basis1));
            const auto v2 = SimdType::load_aligned(&state.at_unsafe(basis2));
            const auto v3 = SimdType::load_aligned(&state.at_unsafe(basis3));
            (inv_sqrt2 * (v1 + plus_i * v3)).store_aligned(&state.at_unsafe(basis0));
            (inv_sqrt2 * (v0 + minus_i * v2)).store_aligned(&state.at_unsafe(basis1));
            (inv_sqrt2 * (plus_i * v1 + v3)).store_aligned(&state.at_unsafe(basis2));
            (inv_sqrt2 * (minus_i * v0 + v2)).store_aligned(&state.at_unsafe(basis3));
        });
}

template <UpdatableStateVector State>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              State& state) {
    if constexpr (supports_gate_simd<State>) {
        const std::uint64_t skip_mask = physical_target_mask | physical_control_mask | control_mask;
        if (can_use_gate_simd(skip_mask, state)) {
            ecr_gate_simd(physical_target_mask,
                          physical_control_mask,
                          control_mask,
                          control_value_mask,
                          state);
            return;
        }
    }
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    Kokkos::parallel_for(
        "ecr_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0,
            state.flat_dim() >>
                std::popcount(physical_target_mask | physical_control_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(
                    it, physical_target_mask | physical_control_mask | control_mask) |
                control_value_mask;
            std::uint64_t basis_1 = basis_0 | physical_control_mask;
            std::uint64_t basis_2 = basis_0 | physical_target_mask;
            std::uint64_t basis_3 = basis_1 | physical_target_mask;

            ComplexType val0 = state.at_unsafe(basis_0);
            ComplexType val1 = state.at_unsafe(basis_1);
            ComplexType val2 = state.at_unsafe(basis_2);
            ComplexType val3 = state.at_unsafe(basis_3);

            ComplexType res0 = (val1 + val3 * ComplexType(0, 1)) * ComplexType(INVERSE_SQRT2());
            ComplexType res1 = (val0 + val2 * ComplexType(0, -1)) * ComplexType(INVERSE_SQRT2());
            ComplexType res2 = (val1 * ComplexType(0, 1) + val3) * ComplexType(INVERSE_SQRT2());
            ComplexType res3 = (val0 * ComplexType(0, -1) + val2) * ComplexType(INVERSE_SQRT2());

            state.at_unsafe(basis_0) = res0;
            state.at_unsafe(basis_1) = res1;
            state.at_unsafe(basis_2) = res2;
            state.at_unsafe(basis_3) = res3;
        });
}

template <UpdatableStateVector State>
void permutation_gate(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& swap_schedule,
                      State& state) {
    using ExecSpace = SpaceType<State::space>;
    for (const auto& pair : swap_schedule) {
        const std::uint64_t src = pair.first;
        const std::uint64_t dst = pair.second;
        const std::uint64_t target_mask = (1ULL << src) | (1ULL << dst);
        Kokkos::parallel_for(
            "permutation_gate",
            Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim() >> 2),
            KOKKOS_LAMBDA(std::uint64_t it) {
                const std::uint64_t basis = insert_zero_at_mask_positions(it, target_mask);
                Kokkos::kokkos_swap(state.at_unsafe(basis | (1ULL << src)),
                                    state.at_unsafe(basis | (1ULL << dst)));
            });
    }
}

// clang-format off
#define INSTANTIATE_FLAT_STATE_OVERLOADS(Func, ...)                                      \
    template void Func<StateVector<Prec, Space>>(__VA_ARGS__, StateVector<Prec, Space>&); \
    template void Func<StateVectorBatched<Prec, Space>>(                                 \
        __VA_ARGS__, StateVectorBatched<Prec, Space>&)

INSTANTIATE_FLAT_STATE_OVERLOADS(swap_gate,                      std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(ecr_gate,                       std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(permutation_gate,               const std::vector<std::pair<std::uint64_t, std::uint64_t>>&);

#undef INSTANTIATE_FLAT_STATE_OVERLOADS
// clang-format on

}  // namespace scaluq::internal
