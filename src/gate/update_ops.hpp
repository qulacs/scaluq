#pragma once

#include <scaluq/operator/pauli_operator.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/types.hpp>

namespace scaluq {
namespace internal {

template <FloatingPoint Fp>
void none_target_dense_matrix_gate(std::uint64_t control_mask,
                                   const Matrix<Fp>& matrix,
                                   StateVector<Fp>& state);

template <FloatingPoint Fp>
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2<Fp>& matrix,
                                  StateVector<Fp>& state);

template <FloatingPoint Fp>
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4<Fp>& matrix,
                                  StateVector<Fp>& state);

template <FloatingPoint Fp>
void single_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state);

template <FloatingPoint Fp>
void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const Matrix<Fp>& matrix,
                                     StateVector<Fp>& state);

template <FloatingPoint Fp>
void multi_target_dense_matrix_gate(std::uint64_t target_mask,
                                    std::uint64_t control_mask,
                                    const Matrix<Fp>& matrix,
                                    StateVector<Fp>& state);

template <FloatingPoint Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state);

template <FloatingPoint Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& mat,
                        StateVector<Fp>& state);

template <FloatingPoint Fp>
inline Matrix2x2<Fp> get_IBMQ_matrix(Fp _theta, Fp _phi, Fp _lambda);

template <FloatingPoint Fp>
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2<Fp>& diag,
                                     StateVector<Fp>& state);

template <FloatingPoint Fp>
inline void i_gate(std::uint64_t, std::uint64_t, StateVector<Fp>&) {}

template <FloatingPoint Fp>
void global_phase_gate(std::uint64_t, std::uint64_t control_mask, Fp angle, StateVector<Fp>& state);

template <FloatingPoint Fp>
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);

template <FloatingPoint Fp>
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);

template <FloatingPoint Fp>
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);

template <FloatingPoint Fp>
inline void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, HADAMARD_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
void one_target_phase_gate(std::uint64_t target_mask,
                           std::uint64_t control_mask,
                           Complex<Fp> phase,
                           StateVector<Fp>& state);

template <FloatingPoint Fp>
inline void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, 1), state);
}

template <FloatingPoint Fp>
inline void sdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(target_mask, control_mask, Complex<Fp>(0, -1), state);
}

template <FloatingPoint Fp>
inline void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask,
        control_mask,
        Complex<Fp>(static_cast<Fp>(INVERSE_SQRT2()), static_cast<Fp>(INVERSE_SQRT2())),
        state);
}

template <FloatingPoint Fp>
inline void tdag_gate(std::uint64_t target_mask,
                      std::uint64_t control_mask,
                      StateVector<Fp>& state) {
    one_target_phase_gate(
        target_mask,
        control_mask,
        Complex<Fp>(static_cast<Fp>(INVERSE_SQRT2()), -static_cast<Fp>(INVERSE_SQRT2())),
        state);
}

template <FloatingPoint Fp>
inline void sqrtx_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_GATE_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
inline void sqrtxdag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_X_DAG_GATE_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
inline void sqrty_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_GATE_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
inline void sqrtydag_gate(std::uint64_t target_mask,
                          std::uint64_t control_mask,
                          StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, SQRT_Y_DAG_GATE_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
inline void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_0_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
inline void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state) {
    one_target_dense_matrix_gate(target_mask, control_mask, PROJ_1_MATRIX<Fp>(), state);
}

template <FloatingPoint Fp>
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp angle,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp lambda,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             Fp theta,
             Fp phi,
             Fp lambda,
             StateVector<Fp>& state);

template <FloatingPoint Fp>
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector<Fp>& state);

template <FloatingPoint Fp>
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix<Fp>& matrix,
                        StateVector<Fp>& state);

template <FloatingPoint Fp>
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix<Fp>& matrix,
                       StateVector<Fp>& state);
}  // namespace internal
}  // namespace scaluq
