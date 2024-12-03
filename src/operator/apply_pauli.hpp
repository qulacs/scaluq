#pragma once

#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"

namespace scaluq::internal {

template <std::floating_point Fp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVector<Fp>& state_vector);
template <std::floating_point Fp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVectorBatched<Fp>& states);
template <std::floating_point Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVector<Fp>& state_vector);
template <std::floating_point Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVectorBatched& states);
template <std::floating_point Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp pcoef,
                          std::vector<Fp> params,
                          StateVectorBatched<Fp>& states);
}  // namespace scaluq::internal
