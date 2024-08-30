#include "apply_pauli.hpp"

#include <Kokkos_Core.hpp>

#include "../constant.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq::internal {
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex coef,
                 StateVector& state_vector) {
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            state_vector.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx = insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
    Complex global_phase = PHASE_M90ROT().val[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        state_vector.dim() >> (std::popcount(control_mask) + 1), KOKKOS_LAMBDA(std::uint64_t i) {
            std::uint64_t basis_0 =
                insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) | control_mask;
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            Complex tmp1 = state_vector._raw[basis_0] * global_phase;
            Complex tmp2 = state_vector._raw[basis_1] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp2 = -tmp2;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp1 = -tmp1;
            state_vector._raw[basis_0] = tmp2 * coef;
            state_vector._raw[basis_1] = tmp1 * coef;
        });
    Kokkos::fence();
}
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex coef,
                          double angle,
                          StateVector& state_vector) {
    std::uint64_t global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex true_angle = angle * coef;
    const Complex cosval = Kokkos::cos(-true_angle / 2);
    const Complex sinval = Kokkos::sin(-true_angle / 2);
    if (bit_flip_mask == 0) {
        const Complex cval_min = cosval - Complex(0, 1) * sinval;
        const Complex cval_pls = cosval + Complex(0, 1) * sinval;
        Kokkos::parallel_for(
            state_vector.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t state_idx = insert_zero_at_mask_positions(i, control_mask) | control_mask;
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
            state_vector.dim() >> (std::popcount(control_mask) + 1), KOKKOS_LAMBDA(std::uint64_t i) {
                std::uint64_t basis_0 =
                    internal::insert_zero_at_mask_positions(i, control_mask | 1ULL << pivot) |
                    control_mask;
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;

                // fetch values
                Complex cval_0 = state_vector._raw[basis_0];
                Complex cval_1 = state_vector._raw[basis_1];

                // set values
                state_vector._raw[basis_0] =
                    cosval * cval_0 +
                    Complex(0, 1) * sinval * cval_1 *
                        PHASE_M90ROT().val[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                state_vector._raw[basis_1] =
                    cosval * cval_1 +
                    Complex(0, 1) * sinval * cval_0 *
                        PHASE_M90ROT().val[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
            });
        Kokkos::fence();
    }
}
}  // namespace scaluq::internal
