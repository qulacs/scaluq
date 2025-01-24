#pragma once

#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/state/state_vector_batched.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <Precision Prec>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Prec>& matrix,
                                   StateVector<Prec>& state);
template <Precision Prec>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Prec>& matrix,
                                   StateVectorBatched<Prec>& states);

template <Precision Prec>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVector<Prec>& state);
template <Precision Prec>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Prec>& matrix,
                                  StateVectorBatched<Prec>& states);

template <Precision Prec>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVector<Prec>& state);
template <Precision Prec>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Prec>& matrix,
                                  StateVectorBatched<Prec>& states);

template <Precision Prec>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec>& matrix,
                                     StateVector<Prec>& state);
template <Precision Prec>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec>& matrix,
                                     StateVectorBatched<Prec>& states);

template <Precision Prec>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec>& matrix,
                                     StateVector<Prec>& state);
template <Precision Prec>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Prec>& matrix,
                                     StateVectorBatched<Prec>& states);

template <Precision Prec>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Prec>& matrix,
                                    StateVector<Prec>& state);
template <Precision Prec>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Prec>& matrix,
                                    StateVectorBatched<Prec>& states);

template <Precision Prec>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec>& matrix,
                       StateVector<Prec>& state);
template <Precision Prec>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec>& matrix,
                       StateVectorBatched<Prec>& states);

template <Precision Prec>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Prec>& mat,
                        StateVector<Prec>& state);
template <Precision Prec>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Prec>& mat,
                        StateVectorBatched<Prec>& states);

template <Precision Prec>
inline Matrix2x2<Prec> get_IBMQ_matrix(Float<Prec> _theta, Float<Prec> _phi, Float<Prec> _lambda);

template <Precision Prec>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVector<Prec>& state);
template <Precision Prec>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Prec>& diag,
                                     StateVectorBatched<Prec>& states);

template <Precision Prec>
inline void i_gate(std::uint64_t, std::uint64_t, StateVector<Prec>&) {}
template <Precision Prec>
inline void i_gate(std::uint64_t, std::uint64_t, StateVectorBatched<Prec>&) {}

template <Precision Prec>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Float<Prec> angle,
                       StateVector<Prec>& state);
template <Precision Prec>
void global_phase_gate(std::uint64_t,
                       std::uint64_t control_mask,
                       Float<Prec> angle,
                       StateVectorBatched<Prec>& states);

template <Precision Prec>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state);
template <Precision Prec>
void x_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states);

template <Precision Prec>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state);
template <Precision Prec>
void y_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states);

template <Precision Prec>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state);
template <Precision Prec>
void z_gate(std::uint64_t target_mask,
            std::uint64_t control_mask,
            StateVectorBatched<Prec>& states);

template <Precision Prec>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void h_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Prec>(), states);
}

template <Precision Prec>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Prec> phase,
                           StateVector<Prec>& state);
template <Precision Prec>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Prec> phase,
                           StateVectorBatched<Prec>& states);

template <Precision Prec>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Prec>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Prec>(0, 1), state);
}
template <Precision Prec>
inline void s_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Prec>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Prec>(0, 1), states);
}

template <Precision Prec>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Prec>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Prec>(0, -1), state);
}
template <Precision Prec>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Prec>& states) {
    one_target_phase_gate(target_mask, control_mask, Complex<Prec>(0, -1), states);
}

template <Precision Prec>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVector<Prec>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2()), state);
}
template <Precision Prec>
inline void t_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched<Prec>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Prec>(INVERSE_SQRT2(), INVERSE_SQRT2()), states);
}

template <Precision Prec>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Prec>& state) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()), state);
}
template <Precision Prec>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVectorBatched<Prec>& states) {
    one_target_phase_gate(
        target_mask, control_mask, Complex<Prec>(INVERSE_SQRT2(), -INVERSE_SQRT2()), states);
}

template <Precision Prec>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Prec>(), states);
}

template <Precision Prec>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void p0_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Prec>(), states);
}

template <Precision Prec>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVector<Prec>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Prec>(), state);
}
template <Precision Prec>
inline void p1_gate(std::uint64_t target_mask,
                    std::uint64_t control_mask,
                    StateVectorBatched<Prec>& states) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Prec>(), states);
}

template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state);
template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states);
template <Precision Prec>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state);
template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states);
template <Precision Prec>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVector<Prec>& state);
template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> angle,
             StateVectorBatched<Prec>& states);
template <Precision Prec>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> pcoef,
             std::vector<Float<Prec>> params,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> lambda,
             StateVector<Prec>& state);
template <Precision Prec>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec>& state);
template <Precision Prec>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVector<Prec>& state);
template <Precision Prec>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Float<Prec> theta,
             Float<Prec> phi,
             Float<Prec> lambda,
             StateVectorBatched<Prec>& states);

template <Precision Prec>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Prec>& state);
template <Precision Prec>
void swap_gate(std::uint64_t target_mask,
               std::uint64_t control_mask,
               StateVectorBatched<Prec>& states);

template <Precision Prec>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Prec>& matrix,
                        StateVector<Prec>& state);
template <Precision Prec>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Prec>& matrix,
                        StateVectorBatched<Prec>& states);

template <Precision Prec>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec>& matrix,
                       StateVector<Prec>& state);
template <Precision Prec>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Prec>& matrix,
                       StateVectorBatched<Prec>& states);

}  // namespace internal
}  // namespace scaluq
