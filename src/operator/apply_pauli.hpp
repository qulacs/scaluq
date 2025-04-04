#pragma once

#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>

namespace scaluq::internal {

template <Precision Prec, ExecutionSpace Space>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t control_value_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVector<Prec, Space>& state_vector);
template <Precision Prec, ExecutionSpace Space>
void apply_pauli(std::uint64_t control_mask,
                 std::uint64_t control_value_mask,
                 std::uint64_t bit_flip_mask,
                 std::uint64_t phase_flip_mask,
                 Complex<Prec> coef,
                 StateVectorBatched<Prec, Space>& states);
template <Precision Prec, ExecutionSpace Space>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVector<Prec, Space>& state_vector);
template <Precision Prec, ExecutionSpace Space>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> angle,
                          StateVectorBatched<Prec, Space>& states);
template <Precision Prec, ExecutionSpace Space>
void apply_pauli_rotation(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          std::uint64_t bit_flip_mask,
                          std::uint64_t phase_flip_mask,
                          Complex<Prec> coef,
                          Float<Prec> pcoef,
                          std::vector<Float<Prec>> params,
                          StateVectorBatched<Prec, Space>& states);
}  // namespace scaluq::internal
