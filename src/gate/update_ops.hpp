#pragma once

#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVector<Fp>& state);
template <std::floating_point Fp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVector<Fp>& state);
template <std::floating_point Fp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVector<Fp>& state);
template <std::floating_point Fp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state);
template <std::floating_point Fp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state);
template <std::floating_point Fp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVector<Fp>& state);
template <std::floating_point Fp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state);
template <std::floating_point Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVector<Fp>& state);
template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
inline Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda);

template <std::floating_point Fp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVector<Fp>& state);
template <std::floating_point Fp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&) {}
template <std::floating_point Fp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Fp>&) {}

template <std::floating_point Fp>
void global_phase_gate(std::uint64_t, std::uint64_t control_mask, Fp angle, StateVector<Fp>& state);
template <std::floating_point Fp>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Fp angle,
                       StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);
template <std::floating_point Fp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);
template <std::floating_point Fp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);
template <std::floating_point Fp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
inline void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVector<Fp>& state);
template <std::floating_point Fp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
inline void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), state);
}
template <std::floating_point Fp>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), states);
}

template <std::floating_point Fp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), state);
}
template <std::floating_point Fp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Fp>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), states);
}

template <std::floating_point Fp>
inline void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}
template <std::floating_point Fp>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Fp>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), INVERSE_SQRT2()), states);
}

template <std::floating_point Fp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}
template <std::floating_point Fp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Fp>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Fp>(INVERSE_SQRT2(), -INVERSE_SQRT2()), states);
}

template <std::floating_point Fp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
inline void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
inline void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), state);
}
template <std::floating_point Fp>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Fp>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), states);
}

template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states);
template <std::floating_point Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states);
template <std::floating_point Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVectorBatched<Fp>& states);
template <std::floating_point Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp pcoef,
             std::vector<Fp> params,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state);
template <std::floating_point Fp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);
template <std::floating_point Fp>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& matrix,
                        StateVector<Fp>& state);
template <std::floating_point Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& matrix,
                        StateVectorBatched<Fp>& states);

template <std::floating_point Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state);
template <std::floating_point Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVectorBatched<Fp>& states);

}  // namespace internal
}  // namespace scaluq
