#pragma once

#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>

namespace scaluq::internal {

template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVector<Fp, Sp>& state_vector);
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Fp> coef,
                 StateVectorBatched<Fp, Sp>& states);
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVector<Fp, Sp>& state_vector);
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp angle,
                          StateVectorBatched<Fp, Sp>& states);
template <std::floating_point Fp, ExecutionSpace Sp>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Fp> coef,
                          Fp pcoef,
                          std::vector<Fp> params,
                          StateVectorBatched<Fp, Sp>& states);
}  // namespace scaluq::internal
