#pragma once

#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>

namespace scaluq::internal {

<<<<<<< HEAD
template <Precision Prec>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVector<Prec>& state_vector);
template <Precision Prec>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVectorBatched<Prec>& states);
template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVector<Prec>& state_vector);
template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVectorBatched<Prec>& states);
template <Precision Prec>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> pcoef,
                          std::vector<Float<Prec>> params,
                          StateVectorBatched<Prec>& states);
=======
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
>>>>>>> set-space
}  // namespace scaluq::internal
