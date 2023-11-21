#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
std::array<Complex, 4> get_IBMQ_matrix(double theta, double phi, double lambda) {
    std::array<Complex, 4> matrix;
    Complex im = Complex(0, 1);
    Complex exp_val1 = std::exp(im * phi);
    Complex exp_val2 = std::exp(im * lambda);
    Complex cos_val = std::cos(theta / 2.);
    Complex sin_val = std::sin(theta / 2.);
    matrix[0] = cos_val;
    matrix[1] = -exp_val2 * sin_val;
    matrix[2] = exp_val1 * sin_val;
    matrix[3] = exp_val1 * exp_val2 * cos_val;
    return matrix;
}

void u_gate(UINT target_qubit_index, std::array<Complex, 4> matrix, StateVector& state) {
    single_qubit_dense_matrix_gate(target_qubit_index, matrix, state);
}
}  // namespace qulacs
