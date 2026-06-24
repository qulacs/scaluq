#include "update_ops.hpp"

namespace scaluq::internal {

namespace {

template <Precision Prec>
Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda) {
    Complex<Prec> exp_val1 = internal::exp(Complex<Prec>(0, _phi));
    Complex<Prec> exp_val2 = internal::exp(Complex<Prec>(0, _lambda));
    Complex<Prec> cos_val = internal::cos(_theta / Float<Prec>{2});
    Complex<Prec> sin_val = internal::sin(_theta / Float<Prec>{2});
    return {
        {{{cos_val, -exp_val2 * sin_val}}, {{exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val}}}};
}

}  // namespace

template <UpdatableStateVector State>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<State::prec> angle,
                       State& state) {
    using ExecSpace = SpaceType<State::space>;
    Complex<State::prec> coef = internal::polar<State::prec>(Float<State::prec>{1}, angle);
    Kokkos::parallel_for(
        "global_phase_gate",
        Kokkos::RangePolicy<ExecSpace>(0, state.flat_dim() >> std::popcount(control_mask)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            state.at_unsafe(insert_zero_at_mask_positions(i, control_mask) | control_value_mask) *=
                coef;
        });
}

template <UpdatableStateVector State>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<State::prec> phase,
                           State& state) {
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "one_target_phase_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state.at_unsafe(i | target_mask) *= phase;
        });
}

template <UpdatableStateVector State>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state) {
    using ExecSpace = SpaceType<State::space>;
    Kokkos::parallel_for(
        "x_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            Kokkos::kokkos_swap(state.at_unsafe(i), state.at_unsafe(i | target_mask));
        });
}

template <UpdatableStateVector State>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    Kokkos::parallel_for(
        "y_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state.at_unsafe(i) *= ComplexType(0, 1);
            state.at_unsafe(i | target_mask) *= ComplexType(0, -1);
            Kokkos::kokkos_swap(state.at_unsafe(i), state.at_unsafe(i | target_mask));
        });
}

template <UpdatableStateVector State>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state) {
    using ExecSpace = SpaceType<State::space>;
    using ComplexType = Complex<State::prec>;
    Kokkos::parallel_for(
        "z_gate",
        Kokkos::RangePolicy<ExecSpace>(
            0, state.flat_dim() >> std::popcount(target_mask | control_mask)),
        KOKKOS_LAMBDA(std::uint64_t it) {
            std::uint64_t i =
                insert_zero_at_mask_positions(it, control_mask | target_mask) | control_value_mask;
            state.at_unsafe(i | target_mask) *= ComplexType(-1, 0);
        });
}

template <UpdatableStateVector State>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state) {
    using ComplexType = Complex<State::prec>;
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    Matrix2x2<State::prec> matrix = {
        {{{cosval, ComplexType(0, -sinval)}}, {{ComplexType(0, -sinval), cosval}}}};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, state);
}

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

template <UpdatableStateVector State>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state) {
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    Matrix2x2<State::prec> matrix = {{{{cosval, -sinval}}, {{sinval, cosval}}}};
    one_target_dense_matrix_gate(target_mask, control_mask, control_value_mask, matrix, state);
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

template <UpdatableStateVector State>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state) {
    using ComplexType = Complex<State::prec>;
    const Float<State::prec> cosval = internal::cos(angle / Float<State::prec>{2});
    const Float<State::prec> sinval = internal::sin(angle / Float<State::prec>{2});
    DiagonalMatrix2x2<State::prec> diag = {ComplexType(cosval, -sinval),
                                           ComplexType(cosval, sinval)};
    one_target_diagonal_matrix_gate(target_mask, control_mask, control_value_mask, diag, state);
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
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> lambda,
             State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          internal::exp(Complex<State::prec>(0, lambda)),
                          state);
}

template <UpdatableStateVector State>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> phi,
             Float<State::prec> lambda,
             State& state) {
    one_target_dense_matrix_gate(
        target_mask,
        control_mask,
        control_value_mask,
        get_IBMQ_matrix<State::prec>(
            static_cast<Float<State::prec>>(Kokkos::numbers::pi / 2), phi, lambda),
        state);
}

template <UpdatableStateVector State>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> theta,
             Float<State::prec> phi,
             Float<State::prec> lambda,
             State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 get_IBMQ_matrix<State::prec>(theta, phi, lambda),
                                 state);
}

template <UpdatableStateVector State>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               State& state) {
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
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              State& state) {
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

INSTANTIATE_FLAT_STATE_OVERLOADS(one_target_phase_gate,          std::uint64_t, std::uint64_t, std::uint64_t, Complex<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(global_phase_gate,              std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(x_gate,                         std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(y_gate,                         std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(z_gate,                         std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(rx_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(ry_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(rz_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(u1_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(u2_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(u3_gate,                        std::uint64_t, std::uint64_t, std::uint64_t, Float<Prec>, Float<Prec>, Float<Prec>);
INSTANTIATE_FLAT_STATE_OVERLOADS(swap_gate,                      std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(ecr_gate,                       std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t);
INSTANTIATE_FLAT_STATE_OVERLOADS(permutation_gate,               const std::vector<std::pair<std::uint64_t, std::uint64_t>>&);

#undef INSTANTIATE_FLAT_STATE_OVERLOADS
// clang-format on

}  // namespace scaluq::internal
