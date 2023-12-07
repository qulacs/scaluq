#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <types.hpp>

namespace qulacs {
static Eigen::MatrixXcd kronecker_product(const Eigen::MatrixXcd& lhs,
                                          const Eigen::MatrixXcd& rhs) {
    Eigen::MatrixXcd result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.cols(); i++) {
        for (int j = 0; j < lhs.rows(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(
    UINT target_qubit_index, const Eigen::MatrixXcd& one_qubit_matrix, UINT qubit_count) {
    const int left_dim = 1ULL << target_qubit_index;
    const int right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return kronecker_product(kronecker_product(right_identity, one_qubit_matrix), left_identity);
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CNOT(UINT control_qubit_index,
                                                         UINT target_qubit_index,
                                                         UINT qubit_count) {
    int dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (int ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CZ(UINT control_qubit_index,
                                                       UINT target_qubit_index,
                                                       UINT qubit_count) {
    int dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (int ind = 0; ind < dim; ++ind) {
        if ((ind & (1ULL << control_qubit_index)) != 0 &&
            (ind & (1ULL << target_qubit_index)) != 0) {
            result(ind, ind) = -1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

#define ASSERT_STATE_NEAR(state, other, eps) \
    ASSERT_PRED_FORMAT3(_assert_state_near, state, other, eps)

static testing::AssertionResult _assert_state_near(const char* state1_name,
                                                   const char* state2_name,
                                                   const char* eps_name,
                                                   const StateVector& state1,
                                                   const StateVector& state2,
                                                   const double eps) {
    if (state1.dim() != state2.dim()) {
        return testing::AssertionFailure()
               << "The dimension is different\nDimension of " << state1_name << " is "
               << state1.dim() << ",\n"
               << "Dimension of " << state2_name << " is " << state2.dim() << ".";
    }

    for (int i = 0; i < state1.dim(); i++) {
        const double real_diff =
            std::fabs(state1.amplitudes()[i].real() - state2.amplitudes()[i].real());
        if (real_diff > eps) {
            return testing::AssertionFailure()
                   << "The difference between " << i << "-th real part of " << state1_name
                   << " and " << state2_name << " is " << real_diff << ", which exceeds " << eps
                   << ", where\n"
                   << state1_name << " evaluates to " << state1.amplitudes()[i].real() << ",\n"
                   << state2_name << " evaluates to " << state2.amplitudes()[i].real() << ", and\n"
                   << eps_name << " evaluates to " << eps << ".";
        }

        const double imag_diff =
            std::fabs(state1.amplitudes()[i].real() - state2.amplitudes()[i].real());
        if (imag_diff > eps) {
            return testing::AssertionFailure()
                   << "The difference between " << i << "-th imaginary part of " << state1_name
                   << " and " << state2_name << " is " << imag_diff << ", which exceeds " << eps
                   << ", where\n"
                   << state1_name << " evaluates to " << state1.amplitudes()[i].imag() << ",\n"
                   << state2_name << " evaluates to " << state2.amplitudes()[i].imag() << ", and\n"
                   << eps_name << " evaluates to " << eps << ".";
        }
    }

    return testing::AssertionSuccess();
}

static Eigen::MatrixXcd make_2x2_matrix(const Eigen::dcomplex a00,
                                        const Eigen::dcomplex a01,
                                        const Eigen::dcomplex a10,
                                        const Eigen::dcomplex a11) {
    Eigen::MatrixXcd m(2, 2);
    m << a00, a01, a10, a11;
    return m;
}

static Eigen::MatrixXcd make_I() { return Eigen::MatrixXcd::Identity(2, 2); }

static Eigen::MatrixXcd make_X() { return make_2x2_matrix(0, 1, 1, 0); }

static Eigen::MatrixXcd make_Y() { return make_2x2_matrix(0, -1.i, 1.i, 0); }

static Eigen::MatrixXcd make_Z() { return make_2x2_matrix(1, 0, 0, -1); }

static Eigen::MatrixXcd make_H() {
    return make_2x2_matrix(1 / sqrt(2.), 1 / sqrt(2.), 1 / sqrt(2.), -1 / sqrt(2.));
}

static Eigen::MatrixXcd make_S() { return make_2x2_matrix(1, 0, 0, 1.i); }

static Eigen::MatrixXcd make_Sdag() { return make_2x2_matrix(1, 0, 0, -1.i); }

static Eigen::MatrixXcd make_T() { return make_2x2_matrix(1, 0, 0, (1. + 1.i) / sqrt(2.)); }

static Eigen::MatrixXcd make_Tdag() { return make_2x2_matrix(1, 0, 0, (1. - 1.i) / sqrt(2.)); }

static Eigen::MatrixXcd make_sqrtX() {
    return make_2x2_matrix(0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i);
}

static Eigen::MatrixXcd make_sqrtY() {
    return make_2x2_matrix(0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i);
}

static Eigen::MatrixXcd make_sqrtXdag() {
    return make_2x2_matrix(0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i);
}

static Eigen::MatrixXcd make_sqrtYdag() {
    return make_2x2_matrix(0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i);
}

static Eigen::MatrixXcd make_P0() { return make_2x2_matrix(1, 0, 0, 0); }

static Eigen::MatrixXcd make_P1() { return make_2x2_matrix(0, 0, 0, 1); }

static Eigen::MatrixXcd make_RX(double angle) {
    return make_2x2_matrix(std::cos(angle / 2),
                           -1i * std::sin(angle / 2),
                           -1i * std::sin(angle / 2),
                           std::cos(angle / 2));
}

static Eigen::MatrixXcd make_RY(double angle) {
    return make_2x2_matrix(
        std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2));
}

static Eigen::MatrixXcd make_RZ(double angle) {
    return make_2x2_matrix(std::exp(-1i * (angle / 2)), 0, 0, std::exp(1i * (angle / 2)));
}

}  // namespace qulacs
