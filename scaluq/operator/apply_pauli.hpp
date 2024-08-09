#pragma once

#include "../state/state_vector.hpp"

namespace scaluq::internal {
void apply_pauli(UINT control_mask,
                 UINT bit_flip_mask,
                 UINT phase_flip_mask,
                 Complex coef,
                 StateVector& state_vector);
void apply_pauli_rotation(UINT control_mask,
                          UINT bit_flip_mask,
                          UINT phase_flip_mask,
                          Complex coef,
                          double angle,
                          StateVector& state_vector);
}  // namespace scaluq::internal
