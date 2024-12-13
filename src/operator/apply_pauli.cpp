#include "apply_pauli.hpp"

#include <scaluq/constant.hpp>

#include "../util/template.hpp"

namespace scaluq::internal {
template <std::floating_point Fp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVector<Fp>& state_vector) {
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            state_vector.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
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
    Complex<Fp> global_phase = PHASE_M90ROT<Fp>()[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        state_vector.dim() >> (std::popcount(control_mask) + 1), KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_mask;
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            Complex<Fp> tmp1 = state_vector._raw[basis_0] * global_phase;
            Complex<Fp> tmp2 = state_vector._raw[basis_1] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp2 = -tmp2;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp1 = -tmp1;
            state_vector._raw[basis_0] = tmp2 * coef;
            state_vector._raw[basis_1] = tmp1 * coef;
        });
    Kokkos::fence();
}
#define FUNC_MACRO(Fp)         \
    template void apply_pauli( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Fp>, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO

template <std::floating_point Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVector<Fp>& state_vector) {
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex<Fp> true_angle = angle * coef;
    const Complex<Fp> cosval = Kokkos::cos(-true_angle / 2);
    const Complex<Fp> sinval = Kokkos::sin(-true_angle / 2);
    if (bit_flip_mask == 0) {
        const Complex<Fp> cval_min = cosval - Complex<Fp>(0, 1) * sinval;
        const Complex<Fp> cval_pls = cosval + Complex<Fp>(0, 1) * sinval;
        Kokkos::parallel_for(
            state_vector.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
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
            state_vector.dim() >> (std::popcount(control_mask) + 1),
            KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_mask;
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;

                // fetch values
                Complex<Fp> cval_0 = state_vector._raw[basis_0];
                Complex<Fp> cval_1 = state_vector._raw[basis_1];

                // set values
                state_vector._raw[basis_0] =
                    cosval * cval_0 +
                    Complex<Fp>(0, 1) * sinval * cval_1 *
                        PHASE_M90ROT<Fp>()[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                state_vector._raw[basis_1] =
                    cosval * cval_1 +
                    Complex<Fp>(0, 1) * sinval * cval_0 *
                        PHASE_M90ROT<Fp>()[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
            });
        Kokkos::fence();
    }
}
#define FUNC_MACRO(Fp)                  \
    template void apply_pauli_rotation( \
        std::uint64_t, std::uint64_t, std::uint64_t, Complex<Fp>, Fp, StateVector<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq::internal
