#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"

namespace scaluq::internal {
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex coef,
                 StateVector& state_vector);
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex coef,
                 StateVectorBatched& states);
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex coef,
                          double angle,
                          StateVector& state_vector);
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex coef,
                          double angle,
                          StateVectorBatched& states);
}  // namespace scaluq::internal
