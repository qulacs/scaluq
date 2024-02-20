#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../constant.hpp"
#include "../operator/pauli_operator.hpp"
#include "../types.hpp"
#include "update_ops.hpp"

namespace qulacs {
void pauli_gate(PauliOperator* pauli, StateVector& state) { pauli->apply_to_state(state); }

void pauli_rotation_gate(PauliOperator* pauli, double angle, StateVector& state) {
    auto [bit_flip_mask_vector, phase_flip_mask_vector] = pauli->get_XZ_mask_representation();
    UINT bit_flip_mask = bit_flip_mask_vector.data_raw()[0];
    UINT phase_flip_mask = phase_flip_mask_vector.data_raw()[0];
    UINT global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    const double cosval = cos(angle / 2);
    const double sinval = sin(angle / 2);
    const Complex coef = pauli->get_coef();
    const auto& amplitudes = state.amplitudes_raw();
    if (bit_flip_mask == 0) {
        Kokkos::parallel_for(
            state.dim(), KOKKOS_LAMBDA(const UINT& state_idx) {
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    amplitudes[state_idx] *= cosval - Complex(0, 1) * sinval;
                } else {
                    amplitudes[state_idx] *= cosval + Complex(0, 1) * sinval;
                }
                amplitudes[state_idx] *= coef;
            });
        return;
    } else {
        const UINT mask = 1 << bit_flip_mask_vector.msb();
        const UINT mask_low = mask - 1;
        const UINT mask_high = ~mask_low;
        Kokkos::parallel_for(
            state.dim(), KOKKOS_LAMBDA(const UINT& state_idx) {
                UINT basis_0 = (state_idx & mask_low) + ((state_idx & mask_high) << 1);
                UINT basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) % 2;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) % 2;

                // fetch values
                Complex cval_0 = amplitudes[basis_0];
                Complex cval_1 = amplitudes[basis_1];

                // set values
                amplitudes[basis_0] =
                    cosval * cval_0 +
                    Complex(0, 1) * sinval * cval_1 *
                        (PHASE_M90ROT()).val[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                amplitudes[basis_1] =
                    cosval * cval_1 +
                    Complex(0, 1) * sinval * cval_0 *
                        (PHASE_M90ROT()).val[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
                amplitudes[basis_0] *= coef;
                amplitudes[basis_1] *= coef;
            });
    }
}

}  // namespace qulacs
