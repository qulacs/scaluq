#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
namespace internal {
matrix_2_2 get_IBMQ_matrix(double theta, double phi, double lambda) {
    Complex exp_val1 = Kokkos::exp(Complex(0, phi));
    Complex exp_val2 = Kokkos::exp(Complex(0, lambda));
    Complex cos_val = Kokkos::cos(theta / 2.);
    Complex sin_val = Kokkos::sin(theta / 2.);
    return {cos_val, -exp_val2 * sin_val, exp_val1 * sin_val, exp_val1 * exp_val2 * cos_val};
}

void u1_gate(UINT target_qubit_index, double lambda, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, get_IBMQ_matrix(0., 0., lambda), state);
}

void u2_gate(UINT target_qubit_index, double phi, double lambda, StateVector& state) {
    single_qubit_dense_matrix_gate(
        target_qubit_index, get_IBMQ_matrix(PI() / 2., phi, lambda), state);
}

void u3_gate(UINT target_qubit_index, double theta, double phi, double lambda, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, get_IBMQ_matrix(theta, phi, lambda), state);
}
}  // namespace internal
}  // namespace qulacs
