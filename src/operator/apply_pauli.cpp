#include "apply_pauli.hpp"

#include <scaluq/constant.hpp>

#include "../util/math.hpp"
#include "../util/template.hpp"

namespace scaluq::internal {
<<<<<<< HEAD
template <Precision Prec>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVector<Prec>& state_vector) {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVector<Fp, Sp>& state_vector) {
>>>>>>> set-space
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Sp>(0, state_vector.dim() >> std::popcount(control_mask)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
        Kokkos::RangePolicy<Sp>(0, state_vector.dim() >> (std::popcount(control_mask) + 1)),
        KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_mask;
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
<<<<<<< HEAD
template <Precision Prec>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVectorBatched<Prec>& states) {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
        Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
            {0, 0}, {states.batch_size(), states.dim() >> (std::popcount(control_mask) + 1)}),
        KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_mask;
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

<<<<<<< HEAD
#define FUNC_MACRO(Prec)       \
    template void apply_pauli( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Prec>, StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Prec)       \
    template void apply_pauli( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Prec>, StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO

template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVector<Prec>& state_vector) {
=======
#define FUNC_MACRO(Fp, Sp)     \
    template void apply_pauli( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Fp>, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Fp, Sp)     \
    template void apply_pauli( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Fp>, StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVector<Fp, Sp>& state_vector) {
>>>>>>> set-space
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> true_angle = angle * coef;
    Complex<Prec> half_angle = true_angle / Float<Prec>{2};
    const Complex<Prec> cosval = internal::cos(-half_angle);
    const Complex<Prec> sinval = internal::sin(-half_angle);
    if (bit_flip_mask == 0) {
        const Complex<Prec> cval_min = cosval - Complex<Prec>(0, 1) * sinval;
        const Complex<Prec> cval_pls = cosval + Complex<Prec>(0, 1) * sinval;
        Kokkos::parallel_for(
            Kokkos::RangePolicy<Sp>(0, state_vector.dim() >> std::popcount(control_mask)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
            Kokkos::RangePolicy<Sp>(0, state_vector.dim() >> (std::popcount(control_mask) + 1)),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_mask;
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
<<<<<<< HEAD
template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVectorBatched<Prec>& states) {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Prec> true_angle = angle * coef;
    const Complex<Prec> cosval = internal::cos(-true_angle / Float<Prec>{2});
    const Complex<Prec> sinval = internal::sin(-true_angle / Float<Prec>{2});
    if (bit_flip_mask == 0) {
        const Complex<Prec> cval_min = cosval - Complex<Prec>(0, 1) * sinval;
        const Complex<Prec> cval_pls = cosval + Complex<Prec>(0, 1) * sinval;
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> std::popcount(control_mask)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t state_idx =
                    insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
            Kokkos::MDRangePolicy<Sp, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim() >> (std::popcount(control_mask) + 1)}),
            KOKKOS_LAMBDA(const std::uint64_t batch_id, const std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_mask;
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
<<<<<<< HEAD
template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> pcoef,
                          std::vector<Float<Prec>> params,
                          StateVectorBatched<Prec>& states) {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp pcoef,
                          std::vector<Fp> params,
                          StateVectorBatched<Fp, Sp>& states) {
>>>>>>> set-space
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    auto team_policy = Kokkos::TeamPolicy<Sp>(Sp(), states.batch_size(), Kokkos::AUTO);
    Kokkos::parallel_for(
        team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<Sp>::member_type& team) {
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
                            insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
                                                control_mask;
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
<<<<<<< HEAD
#define FUNC_MACRO(Prec)                              \
    template void apply_pauli_rotation(std::uint64_t, \
                                       std::uint64_t, \
                                       std::uint64_t, \
                                       Complex<Prec>, \
                                       Float<Prec>,   \
                                       StateVector<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Prec)                              \
    template void apply_pauli_rotation(std::uint64_t, \
                                       std::uint64_t, \
                                       std::uint64_t, \
                                       Complex<Prec>, \
                                       Float<Prec>,   \
                                       StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Prec)                                         \
    template void apply_pauli_rotation(std::uint64_t,            \
                                       std::uint64_t,            \
                                       std::uint64_t,            \
                                       Complex<Prec>,            \
                                       Float<Prec>,              \
                                       std::vector<Float<Prec>>, \
                                       StateVectorBatched<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
=======
#define FUNC_MACRO(Fp, Sp)              \
    template void apply_pauli_rotation( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Fp>, Fp, StateVector<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Fp, Sp)                            \
    template void apply_pauli_rotation(std::uint64_t, \
                                       std::uint64_t, \
                                       std::uint64_t, \
                                       Complex<Fp>,   \
                                       Fp,            \
                                       StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
#undef FUNC_MACRO
#define FUNC_MACRO(Fp, Sp)                              \
    template void apply_pauli_rotation(std::uint64_t,   \
                                       std::uint64_t,   \
                                       std::uint64_t,   \
                                       Complex<Fp>,     \
                                       Fp,              \
                                       std::vector<Fp>, \
                                       StateVectorBatched<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
>>>>>>> set-space
#undef FUNC_MACRO
}  // namespace scaluq::internal
