#pragma once

#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   std::uint64_t control_value_mask,
                                   const Matrix<Prec, Space>& matrix,
                                   StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  std::uint64_t control_value_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const Matrix<Prec, Space>& matrix,
                                     StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    std::uint64_t control_value_mask,
                                    const Matrix<Prec, Space>& matrix,
                                    StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& mat,
                        StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& mat,
                        StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
inline Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda);

template <Precision Prec, ExecutionSpace Space>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     std::uint64_t control_value_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
inline void i_gate(std::uint64_t, std::uint64_t, std::uint64_t, StateVector<Prec, Space>&) {}
template <Precision Prec, ExecutionSpace Space>
inline void i_gate(std::uint64_t, std::uint64_t, std::uint64_t, StateVectorBatched<Prec, Space>&) {}

template <Precision Prec, ExecutionSpace Space>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       Float<Prec> angle,
                       StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            std::uint64_t control_value_mask,
            StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, HADAMARD_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, HADAMARD_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           std::uint64_t control_value_mask,
                           Complex<Prec> phase,
                           StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVector<Prec, Space>& state) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<Prec>(0, 1), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVectorBatched<Prec, Space>& states) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<Prec>(0, 1), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      StateVector<Prec, Space>& state) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<Prec>(0, -1), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      StateVectorBatched<Prec, Space>& states) {
    one_target_phase_gate(
        target_mask, control_mask, control_value_mask, Complex<Prec>(0, -1), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVector<Prec, Space>& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2()),
                          state);
}
template <Precision Prec, ExecutionSpace Space>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   std::uint64_t control_value_mask,
                   StateVectorBatched<Prec, Space>& states) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2()),
                          states);
}

template <Precision Prec, ExecutionSpace Space>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      StateVector<Prec, Space>& state) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()),
                          state);
}
template <Precision Prec, ExecutionSpace Space>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      std::uint64_t control_value_mask,
                      StateVectorBatched<Prec, Space>& states) {
    one_target_phase_gate(target_mask,
                          control_mask,
                          control_value_mask,
                          Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()),
                          states);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_DAG_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_X_DAG_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_DAG_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, SQRT_Y_DAG_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_0_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_0_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    StateVector<Prec, Space>& state) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_1_MATRIX<Prec>(), state);
}
template <Precision Prec, ExecutionSpace Space>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    std::uint64_t control_value_mask,
                    StateVectorBatched<Prec, Space>& states) {
    one_target_dense_matrix_gate(
        target_mask, control_mask, control_value_mask, PROJ_1_MATRIX<Prec>(), states);
}

template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states);
template <Precision Prec, ExecutionSpace Space>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states);
template <Precision Prec, ExecutionSpace Space>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec, Space>& states);
template <Precision Prec, ExecutionSpace Space>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             std::uint64_t control_value_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& matrix,
                        StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        const SparseMatrix<Prec, Space>& matrix,
                        StateVectorBatched<Prec, Space>& states);

template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVector<Prec, Space>& state);
template <Precision Prec, ExecutionSpace Space>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       std::uint64_t control_value_mask,
                       const Matrix<Prec, Space>& matrix,
                       StateVectorBatched<Prec, Space>& states);

}  // namespace internal
}  // namespace scaluq
