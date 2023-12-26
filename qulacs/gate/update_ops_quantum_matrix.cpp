#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
matrix_2_2 get_IBMQ_matrix(double theta, double phi, double lambda) {
    matrix_2_2 matrix;
    Complex im = Complex(0, 1);
    Complex exp_val1 = Kokkos::exp(im * phi);
    Complex exp_val2 = Kokkos::exp(im * lambda);
    Complex cos_val = Kokkos::cos(theta / 2.);
    Complex sin_val = Kokkos::sin(theta / 2.);
    matrix.val[0][0] = cos_val;
    matrix.val[0][1] = -exp_val2 * sin_val;
    matrix.val[1][0] = exp_val1 * sin_val;
    matrix.val[1][1] = exp_val1 * exp_val2 * cos_val;
    return matrix;
}

void u_gate(UINT target_qubit_index, matrix_2_2 matrix, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}
}  // namespace qulacs
