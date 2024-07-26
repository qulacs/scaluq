#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
matrix_2_2 get_IBMQ_matrix(double theta, double phi, double lambda) {
    Complex exp_val1 = Kokkos::exp(Complex(0, phi));
    Complex exp_val2 = Kokkos::exp(Complex(0, lambda));
    Complex cos_val = Kokkos::cos(theta / 2.);
    Complex sin_val = Kokkos::sin(theta / 2.);
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}

void u1_gate(UINT target_mask, UINT control_mask, double lambda, StateVector& state) {
    Complex exp_val = Kokkos::exp(Complex(0, lambda));
    Kokkos::parallel_for(
        state.dim() >> (std::popcount(target_mask | control_mask)), KOKKOS_LAMBDA(UINT it) {
            UINT i = internal::insert_zero_at_mask_positions(it, target_mask | control_mask) |
                     control_mask;
            state._raw[i | target_mask] *= exp_val;
        });
    Kokkos::fence();
}

void u2_gate(UINT target_mask, UINT control_mask, double phi, double lambda, StateVector& state) {
    single_qubit_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(PI() / 2., phi, lambda), state);
}

void u3_gate(UINT target_mask,
             UINT control_mask,
             double theta,
             double phi,
             double lambda,
             StateVector& state) {
    single_qubit_dense_matrix_gate(
        target_mask, control_mask, get_IBMQ_matrix(theta, phi, lambda), state);
}
}  // namespace internal
}  // namespace scaluq
