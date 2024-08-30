
#pragma once

#include "../operator/pauli_operator.hpp"
#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace scaluq {
namespace internal {
void i_gate(UINT target_mask, UINT control_mask, StateVector& state);

void global_phase_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state);

void x_gate(UINT target_mask, UINT control_mask, StateVector& state);

void y_gate(UINT target_mask, UINT control_mask, StateVector& state);

void z_gate(UINT target_mask, UINT control_mask, StateVector& state);

void h_gate(UINT target_mask, UINT control_mask, StateVector& state);

void s_gate(UINT target_mask, UINT control_mask, StateVector& state);

void sdag_gate(UINT target_mask, UINT control_mask, StateVector& state);

void t_gate(UINT target_mask, UINT control_mask, StateVector& state);

void tdag_gate(UINT target_mask, UINT control_mask, StateVector& state);

void sqrtx_gate(UINT target_mask, UINT control_mask, StateVector& state);

void sqrtxdag_gate(UINT target_mask, UINT control_mask, StateVector& state);

void sqrty_gate(UINT target_mask, UINT control_mask, StateVector& state);

void sqrtydag_gate(UINT target_mask, UINT control_mask, StateVector& state);

void p0_gate(UINT target_mask, UINT control_mask, StateVector& state);

void p1_gate(UINT target_mask, UINT control_mask, StateVector& state);

void rx_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state);

void ry_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state);

void rz_gate(UINT target_mask, UINT control_mask, double angle, StateVector& state);

matrix_2_2 get_IBMQ_matrix(double _theta, double _phi, double _lambda);

void one_target_dense_matrix_gate(UINT target_mask,
                                  UINT control_mask,
                                  const matrix_2_2& matrix,
                                  StateVector& state);

void two_target_dense_matrix_gate(UINT target_mask,
                                  UINT control_mask,
                                  const matrix_4_4& matrix,
                                  StateVector& state);

void one_target_diagonal_matrix_gate(UINT target_mask,
                                     UINT control_mask,
                                     const diagonal_matrix_2_2& diag,
                                     StateVector& state);

void u1_gate(UINT target_mask, UINT control_mask, double lambda, StateVector& state);

void u2_gate(UINT target_mask, UINT control_mask, double phi, double lambda, StateVector& state);

void u3_gate(UINT target_mask,
             UINT control_mask,
             double theta,
             double phi,
             double lambda,
             StateVector& state);

void swap_gate(UINT target_mask, UINT control_mask, StateVector& state);
}  // namespace internal
}  // namespace scaluq
