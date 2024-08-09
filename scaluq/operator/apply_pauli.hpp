#pragma once

namespace scaluq::internal {
void apply_pauli(UINT control_mask, UINT bit_flip_mask, UINT phase_flip_mask, StateVector& state);
void apply_pauli_rotation(
    UINT control_mask, UINT bit_flip_mask, UINT phase_flip_mask, double angle, StateVector& state);
}  // namespace scaluq::internal
