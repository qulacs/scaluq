#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../constant.hpp"
#include "../operator/pauli_operator.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {

// まだ
void pauli_gate(UINT control_mask, const PauliOperator& pauli, StateVector& state) {
    pauli.apply_to_state(state);
}

// まだ
void pauli_rotation_gate(UINT control_mask,
                         const PauliOperator& pauli,
                         double angle,
                         StateVector& state) {
    auto [bit_flip_mask_vector, phase_flip_mask_vector] = pauli.get_XZ_mask_representation();
    UINT bit_flip_mask = internal::BitVector(bit_flip_mask_vector).data_raw()[0];
    UINT phase_flip_mask = internal::BitVector(phase_flip_mask_vector).data_raw()[0];
    UINT global_phase_90_rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    Complex true_angle = angle * pauli.get_coef();
    const Complex cosval = Kokkos::cos(-true_angle / 2);
    const Complex sinval = Kokkos::sin(-true_angle / 2);
    if (bit_flip_mask == 0) {
        const Complex cval_min = cosval - Complex(0, 1) * sinval;
        const Complex cval_pls = cosval + Complex(0, 1) * sinval;
        Kokkos::parallel_for(
            state.dim(), KOKKOS_LAMBDA(UINT state_idx) {
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) {
                    state._raw[state_idx] *= cval_min;
                } else {
                    state._raw[state_idx] *= cval_pls;
                }
            });
        Kokkos::fence();
        return;
    } else {
        const UINT insert_idx = internal::BitVector(bit_flip_mask_vector).msb();
        Kokkos::parallel_for(
            state.dim() >> 1, KOKKOS_LAMBDA(UINT state_idx) {
                UINT basis_0 = internal::insert_zero_to_basis_index(state_idx, insert_idx);
                UINT basis_1 = basis_0 ^ bit_flip_mask;

                int bit_parity_0 = Kokkos::popcount(basis_0 & phase_flip_mask) & 1;
                int bit_parity_1 = Kokkos::popcount(basis_1 & phase_flip_mask) & 1;

                // fetch values
                Complex cval_0 = state._raw[basis_0];
                Complex cval_1 = state._raw[basis_1];

                // set values
                state._raw[basis_0] =
                    cosval * cval_0 +
                    Complex(0, 1) * sinval * cval_1 *
                        PHASE_M90ROT().val[(global_phase_90_rot_count + bit_parity_0 * 2) % 4];
                state._raw[basis_1] =
                    cosval * cval_1 +
                    Complex(0, 1) * sinval * cval_0 *
                        PHASE_M90ROT().val[(global_phase_90_rot_count + bit_parity_1 * 2) % 4];
            });
        Kokkos::fence();
    }
}

}  // namespace internal
}  // namespace scaluq
