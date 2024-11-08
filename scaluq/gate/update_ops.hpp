
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {
void i_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void i_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void global_phase_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       double angle,
                       StateVector& state);
void global_phase_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       double angle,
                       StateVectorBatched& states);

void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void x_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void y_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void z_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void h_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void s_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void sdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void sdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void t_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void tdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void tdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void sqrtx_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void sqrtx_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void sqrtxdag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void sqrtxdag_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched& states);

void sqrty_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void sqrty_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void sqrtydag_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void sqrtydag_gate(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   StateVectorBatched& states);

void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void p0_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void p1_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state);
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVectorBatched& states);
void rx_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double pcoef,
             std::vector<double> params,
             StateVectorBatched& states);

void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state);
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVectorBatched& states);
void ry_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double pcoef,
             std::vector<double> params,
             StateVectorBatched& states);

void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVector& state);
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double angle,
             StateVectorBatched& states);
void rz_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double pcoef,
             std::vector<double> params,
             StateVectorBatched& states);

Matrix2x2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2& matrix,
                                  StateVector& state);
void one_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix2x2& matrix,
                                  StateVectorBatched& states);

void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4& matrix,
                                  StateVector& state);
void two_target_dense_matrix_gate(std::uint64_t target_mask,
                                  std::uint64_t control_mask,
                                  const Matrix4x4& matrix,
                                  StateVectorBatched& states);

void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2& diag,
                                     StateVector& state);
void one_target_diagonal_matrix_gate(std::uint64_t target_mask,
                                     std::uint64_t control_mask,
                                     const DiagonalMatrix2x2& diag,
                                     StateVectorBatched& states);

void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double lambda,
             StateVector& state);
void u1_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double lambda,
             StateVectorBatched& states);

void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double phi,
             double lambda,
             StateVector& state);
void u2_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double phi,
             double lambda,
             StateVectorBatched& states);

void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double theta,
             double phi,
             double lambda,
             StateVector& state);
void u3_gate(std::uint64_t target_mask,
             std::uint64_t control_mask,
             double theta,
             double phi,
             double lambda,
             StateVectorBatched& states);

void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVector& state);
void swap_gate(std::uint64_t target_mask, std::uint64_t control_mask, StateVectorBatched& states);

void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix& matrix,
                        StateVector& state);
void sparse_matrix_gate(std::uint64_t target_mask,
                        std::uint64_t control_mask,
                        const SparseMatrix& matrix,
                        StateVectorBatched& state);

void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix& matrix,
                       StateVector& state);
void dense_matrix_gate(std::uint64_t target_mask,
                       std::uint64_t control_mask,
                       const Matrix& matrix,
                       StateVectorBatched& state);
}  // namespace internal
}  // namespace scaluq
