#include "apply_pauli.hpp"

#include <scaluq/constant.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"

namespace scaluq::internal {
template <>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t control_value_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVector<Prec, Space>& state_vector) {
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            "apply_pauli",
            Kokkos::RangePolicy<SpaceType<Space>>(
                0, state_vector.dim() >> std::popcount(control_mask)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    state_vector._raw[state_idx] *= -coef;
                } else {
                    state_vector._raw[state_idx] *= coef;
                }
            });
        Kokkos::fence();
        return;
    }
    std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> global_phase = PHASE_M90ROT<Prec>()[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        "apply_pauli",
        Kokkos::RangePolicy<SpaceType<Space>>(
            0, state_vector.dim() >> (std::popcount(control_mask) + 1)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_value_mask;
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            Complex<Prec> tmp1 = state_vector._raw[basis_0] * global_phase;
            Complex<Prec> tmp2 = state_vector._raw[basis_1] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp2 = -tmp2;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp1 = -tmp1;
            state_vector._raw[basis_0] = tmp2 * coef;
            state_vector._raw[basis_1] = tmp1 * coef;
        });
    Kokkos::fence();
}

template <>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t control_value_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVectorBatched<Prec, Space>& states) {
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            "apply_pauli",
            Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    states._raw(batch_id, state_idx) *= -coef;
                } else {
                    states._raw(batch_id, state_idx) *= coef;
                }
            });
        Kokkos::fence();
        return;
    }
    std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> global_phase = PHASE_M90ROT<Prec>()[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        "apply_pauli",
        Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> (std::popcount(control_mask) + 1)}),
        KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_value_mask;
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            Complex<Prec> tmp1 = states._raw(batch_id, basis_0) * global_phase;
            Complex<Prec> tmp2 = states._raw(batch_id, basis_1) * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp2 = -tmp2;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp1 = -tmp1;
            states._raw(batch_id, basis_0) = tmp2 * coef;
            states._raw(batch_id, basis_1) = tmp1 * coef;
        });
    Kokkos::fence();
}

template <>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVector<Prec, Space>& state_vector) {
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> true_angle = angle * coef;
    Complex<Prec> half_angle = true_angle / Float<Prec>{2};
    const Complex<Prec> cosval = internal::cos(-half_angle);
    const Complex<Prec> sinval = internal::sin(-half_angle);
    if (bit_flip_mask == 0) {
        const Complex<Prec> cval_min = cosval - Complex<Prec>(0, 1) * sinval;
        const Complex<Prec> cval_pls = cosval + Complex<Prec>(0, 1) * sinval;
        Kokkos::parallel_for(
            "apply_pauli_rotation",
            Kokkos::RangePolicy<SpaceType<Space>>(
                0, state_vector.dim() >> std::popcount(control_mask)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    state_vector._raw[state_idx] *= cval_min;
                } else {
                    state_vector._raw[state_idx] *= cval_pls;
                }
            });
        Kokkos::fence();
        return;
    } else {
        std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
        Kokkos::parallel_for(
            "apply_pauli_rotation",
            Kokkos::RangePolicy<SpaceType<Space>>(
                0, state_vector.dim() >> (std::popcount(control_mask) + 1)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_value_mask;
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;

                // fetch values
                Complex<Prec> cval_0 = state_vector._raw[basis_0];
                Complex<Prec> cval_1 = state_vector._raw[basis_1];

                // set values
                state_vector._raw[basis_0] =
                    cosval * cval_0 +
                    Complex<Prec>(0, 1) * sinval * cval_1 *
                        PHASE_M90ROT<Prec>()[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                state_vector._raw[basis_1] =
                    cosval * cval_1 +
                    Complex<Prec>(0, 1) * sinval * cval_0 *
                        PHASE_M90ROT<Prec>()[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
            });
        Kokkos::fence();
    }
}
template <>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVectorBatched<Prec, Space>& states) {
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> true_angle = angle * coef;
    const Complex<Prec> cosval = internal::cos(-true_angle / Float<Prec>{2});
    const Complex<Prec> sinval = internal::sin(-true_angle / Float<Prec>{2});
    if (bit_flip_mask == 0) {
        const Complex<Prec> cval_min = cosval - Complex<Prec>(0, 1) * sinval;
        const Complex<Prec> cval_pls = cosval + Complex<Prec>(0, 1) * sinval;
        Kokkos::parallel_for(
            "apply_pauli_rotation",
            Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    states._raw(batch_id, state_idx) *= cval_min;
                } else {
                    states._raw(batch_id, state_idx) *= cval_pls;
                }
            });
        Kokkos::fence();
        return;
    } else {
        std::uint64_t pivot = sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
        Kokkos::parallel_for(
            "apply_pauli_rotation",
            Kokkos::MDRangePolicy<SpaceType<Space>, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> (std::popcount(control_mask) + 1)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_value_mask;
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;

                // fetch values
                Complex<Prec> cval_0 = states._raw(batch_id, basis_0);
                Complex<Prec> cval_1 = states._raw(batch_id, basis_1);

                // set values
                states._raw(batch_id, basis_0) =
                    cosval * cval_0 +
                    Complex<Prec>(0, 1) * sinval * cval_1 *
                        PHASE_M90ROT<Prec>()[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                states._raw(batch_id, basis_1) =
                    cosval * cval_1 +
                    Complex<Prec>(0, 1) * sinval * cval_0 *
                        PHASE_M90ROT<Prec>()[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
            });
        Kokkos::fence();
    }
}
template <>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> pcoef,
                          std::vector<Float<Prec>> params,
                          StateVectorBatched<Prec, Space>& states) {
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    auto team_policy =
        Kokkos::TeamPolicy<SpaceType<Space>>(SpaceType<Space>(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        "apply_pauli_rotation",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<SpaceType<Space>>::member_type& team) {
            const std::uint64_t batch_id = team.league_rank();
            const Float<Prec> angle = pcoef * params[batch_id];
            const Complex<Prec> true_angle = angle * coef;
            const Complex<Prec> cosval = internal::cos(-true_angle / Float<Prec>{2});
            const Complex<Prec> sinval = internal::sin(-true_angle / Float<Prec>{2});
            if (bit_flip_mask == 0) {
                const Complex<Prec> cval_min = cosval - Complex<Prec>(0, 1) * sinval;
                const Complex<Prec> cval_pls = cosval + Complex<Prec>(0, 1) * sinval;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, states.dim() >> std::popcount(control_mask)),
                    [&](const std::uint64_t i) {
                        std::uint64_t state_idx =
                            insert_zero_at_mask_positions(i, control_mask) | control_value_mask;
                        if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                            states._raw(batch_id, state_idx) *= cval_min;
                        } else {
                            states._raw(batch_id, state_idx) *= cval_pls;
                        }
                    });
            } else {
                std::uint64_t pivot =
                    sizeof(std::uint64_t) * 8 - std::countl_zero(bit_flip_mask) - 1;
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,
                                            states.dim() >> (std::popcount(control_mask) + 1)),
                    [&](const std::uint64_t i) {
                        std::uint64_t basis_0 = internal::insert_zero_at_mask_positions(
                                                    i, control_mask | 1ULL << pivot) |
                                                control_value_mask;
                        std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                        int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                        int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;
                        Complex<Prec> cval_0 = states._raw(batch_id, basis_0);
                        Complex<Prec> cval_1 = states._raw(batch_id, basis_1);
                        states._raw(batch_id, basis_0) =
                            cosval * cval_0 +
                            Complex<Prec>(0, 1) * sinval * cval_1 *
                                PHASE_M90ROT<
                                    Prec>()[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                        states._raw(batch_id, basis_1) =
                            cosval * cval_1 +
                            Complex<Prec>(0, 1) * sinval * cval_0 *
                                PHASE_M90ROT<
                                    Prec>()[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
                    });
            }
        });
    Kokkos::fence();
}
}  // namespace scaluq::internal
