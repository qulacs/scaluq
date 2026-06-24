#pragma once

#include <concepts>
#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <class State>
concept UpdatableStateVector = requires(State& state, std::uint64_t index) {
    { State::prec } -> std::convertible_to<Precision>;
    { State::space } -> std::convertible_to<ExecutionSpace>;
    typename State::RawView;
    { state.n_qubits() } -> std::convertible_to<std::uint64_t>;
    { state.flat_dim() } -> std::convertible_to<std::uint64_t>;
    { state.copy() } -> std::same_as<State>;
    state.load(state);
    state.at_unsafe(index);
};

template <UpdatableStateVector State>
void zero_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   Complex<State::prec> matrix,
                                   State& state);

template <UpdatableStateVector State>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<State::prec>& matrix,
                                  State& state);

template <UpdatableStateVector State>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<State::prec>& matrix,
                                  State& state);

template <UpdatableStateVector State>
void multi_dense_matrix_gate(std::uint64_t target_mask,
                             std::uint64_t control_mask,
                             std::uint64_t control_value_mask,
                             const Matrix<State::prec, State::space>& matrix,
                             State& state);

template <UpdatableStateVector State>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<State::prec, State::space>& mat,
                        State& state);

template <UpdatableStateVector State>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<State::prec>& diag,
                                     State& state);

template <UpdatableStateVector State>
inline void i_gate(std::uint64_t, std::uint64_t, std::uint64_t, State&) {}

template <UpdatableStateVector State>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<State::prec> angle,
                       State& state);

template <UpdatableStateVector State>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state);

template <UpdatableStateVector State>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state);

template <UpdatableStateVector State>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            State& state);

template <UpdatableStateVector State>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, HADAMARD_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<State::prec> phase,
                           State& state);

template <UpdatableStateVector State>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<State::prec>(0, 1), state);
}

template <UpdatableStateVector State>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      State& state) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<State::prec>(0, -1), state);
}

template <UpdatableStateVector State>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<State::prec>(INVERSE_SQRT2(), INVERSE_SQRT2()),
                          state);
}

template <UpdatableStateVector State>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      State& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<State::prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()),
                          state);
}

template <UpdatableStateVector State>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_GATE_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 SQRT_X_DAG_GATE_MATRIX<State::prec>(),
                                 state);
}

template <UpdatableStateVector State>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_GATE_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          State& state) {
    one_target_dense_matrix_gate(target_mask,
                                 control_mask,
                                 control_value_mask,
                                 SQRT_Y_DAG_GATE_MATRIX<State::prec>(),
                                 state);
}

template <UpdatableStateVector State>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_0_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    State& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_1_MATRIX<State::prec>(), state);
}

template <UpdatableStateVector State>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state);
template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <UpdatableStateVector State>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state);
template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <UpdatableStateVector State>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> angle,
             State& state);
template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             const Kokkos::View<Float<Prec>*, SpaceType<Space>>& params,
             StateVectorBatched<Prec, Space>& states);

template <UpdatableStateVector State>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> lambda,
             State& state);

template <UpdatableStateVector State>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> phi,
             Float<State::prec> lambda,
             State& state);

template <UpdatableStateVector State>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<State::prec> theta,
             Float<State::prec> phi,
             Float<State::prec> lambda,
             State& state);

template <UpdatableStateVector State>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               State& state);

template <UpdatableStateVector State>
void ecr_gate(std::uint64_t physical_target_mask,
              std::uint64_t physical_control_mask,
              std::uint64_t control_mask,
              std::uint64_t control_value_mask,
              State& state);
template <UpdatableStateVector State>
void permutation_gate(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& swap_schedule,
                      State& state);

}  // namespace internal
}  // namespace scaluq
