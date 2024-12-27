#pragma once

#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>

namespace scaluq::internal {

template <FloatingPoint Fp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVector<Fp>& state_vector);
template <FloatingPoint Fp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVectorBatched<Fp>& states);
template <FloatingPoint Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVector<Fp>& state_vector);
template <FloatingPoint Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVectorBatched<Fp>& states);
template <FloatingPoint Fp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp pcoef,
                          std::vector<Fp> params,
                          StateVectorBatched<Fp>& states);
}  // namespace scaluq::internal
