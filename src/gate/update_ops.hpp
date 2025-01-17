#pragma once

#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <std::floating_point Fp, ExecutionSpace Sp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
inline Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda);

template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVector<Fp, Sp>&) {}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp, Sp>&) {}

template <std::floating_point Fp, ExecutionSpace Sp>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Fp, Sp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp, Sp>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp, Sp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Fp, Sp>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Fp, Sp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp, Sp>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2()), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp, Sp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Fp, Sp>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2()), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVector<Fp, Sp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), state);
}
template <std::floating_point Fp, ExecutionSpace Sp>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Fp, Sp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), states);
}

template <std::floating_point Fp, ExecutionSpace Sp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states);
template <std::floating_point Fp, ExecutionSpace Sp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states);
template <std::floating_point Fp, ExecutionSpace Sp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp, Sp>& states);
template <std::floating_point Fp, ExecutionSpace Sp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp, Sp>& states);

template <std::floating_point Fp, ExecutionSpace Sp>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp, Sp>& state);
template <std::floating_point Fp, ExecutionSpace Sp>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Fp, Sp>& states);

}  // namespace internal
}  // namespace scaluq
