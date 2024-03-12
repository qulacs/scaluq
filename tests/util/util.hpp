#pragma once

#include <Eigen/Core>
#include <complex>
using namespace std::complex_literals;

#include <types.hpp>
#include <util/random.hpp>
using namespace qulacs;

#define _CHECK_GT(val1, val2) _check_gt(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
inline std::string _check_gt(
    T val1, T val2, std::string val1_name, std::string val2_name, std::string file, UINT line) {
    if (val1 > val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") > (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

inline Eigen::MatrixXcd kronecker_product(const Eigen::MatrixXcd& lhs,
                                          const Eigen::MatrixXcd& rhs) {
    Eigen::MatrixXcd result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

// obtain single dense matrix
inline Eigen::MatrixXcd get_eigen_matrix_single_Pauli(UINT pauli_id) {
    Eigen::MatrixXcd mat(2, 2);
    if (pauli_id == 0)
        mat << 1, 0, 0, 1;
    else if (pauli_id == 1)
        mat << 0, 1, 1, 0;
    else if (pauli_id == 2)
        mat << 0, -1.i, 1.i, 0;
    else if (pauli_id == 3)
        mat << 1, 0, 0, -1;
    return mat;
}
inline Eigen::MatrixXcd get_eigen_matrix_random_single_qubit_unitary() {
    Eigen::MatrixXcd Identity, X, Y, Z;
    Identity = get_eigen_matrix_single_Pauli(0);
    X = get_eigen_matrix_single_Pauli(1);
    Y = get_eigen_matrix_single_Pauli(2);
    Z = get_eigen_matrix_single_Pauli(3);

    double icoef, xcoef, ycoef, zcoef, norm;
    Random random;
    icoef = random.uniform();
    xcoef = random.uniform();
    ycoef = random.uniform();
    zcoef = random.uniform();
    norm = sqrt(icoef * icoef + xcoef + xcoef + ycoef * ycoef + zcoef * zcoef);
    icoef /= norm;
    xcoef /= norm;
    ycoef /= norm;
    zcoef /= norm;
    return icoef * Identity + 1.i * xcoef * X + 1.i * ycoef * Y + 1.i * zcoef * Z;
}
inline Eigen::VectorXcd get_eigen_diagonal_matrix_random_multi_qubit_unitary(UINT qubit_count) {
    UINT dim = (1ULL) << qubit_count;
    auto vec = Eigen::VectorXcd(dim);
    Random random;
    for (UINT i = 0; i < dim; ++i) {
        double angle = random.uniform() * 2 * 3.14159;
        vec[i] = cos(angle) + 1.i * sin(angle);
    }
    return vec;
}

inline Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(
    UINT target_qubit_index, const Eigen::MatrixXcd& one_qubit_matrix, UINT qubit_count) {
    const UINT left_dim = 1ULL << target_qubit_index;
    const UINT right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return kronecker_product(kronecker_product(right_identity, one_qubit_matrix), left_identity);
}

// get expanded matrix
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<UINT> pauli_ids) {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(1, 1);
    for (UINT i = 0; i < pauli_ids.size(); ++i) {
        result = kronecker_product(get_eigen_matrix_single_Pauli(pauli_ids[i]), result).eval();
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<UINT> index_list,
                                                          std::vector<UINT> pauli_list,
                                                          UINT qubit_count) {
    std::vector<UINT> whole_pauli_ids(qubit_count, 0);
    for (UINT i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli(whole_pauli_ids);
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_CNOT(UINT control_qubit_index,
                                                         UINT target_qubit_index,
                                                         UINT qubit_count) {
    UINT dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (UINT ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_CZ(UINT control_qubit_index,
                                                       UINT target_qubit_index,
                                                       UINT qubit_count) {
    UINT dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (UINT ind = 0; ind < dim; ++ind) {
        if ((ind & (1ULL << control_qubit_index)) != 0 &&
            (ind & (1ULL << target_qubit_index)) != 0) {
            result(ind, ind) = -1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_SWAP(UINT target_qubit_index1,
                                                         UINT target_qubit_index2,
                                                         UINT qubit_count) {
    UINT dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (UINT ind = 0; ind < dim; ++ind) {
        bool flag1, flag2;
        flag1 = (ind & (1ULL << target_qubit_index1)) != 0;
        flag2 = (ind & (1ULL << target_qubit_index2)) != 0;
        if (flag1 ^ flag2) {
            result(ind, ind ^ (1ULL << target_qubit_index1) ^ (1ULL << target_qubit_index2)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

inline Eigen::MatrixXcd make_2x2_matrix(const Eigen::dcomplex a00,
                                        const Eigen::dcomplex a01,
                                        const Eigen::dcomplex a10,
                                        const Eigen::dcomplex a11) {
    Eigen::MatrixXcd m(2, 2);
    m << a00, a01, a10, a11;
    return m;
}

inline Eigen::MatrixXcd make_I() { return Eigen::MatrixXcd::Identity(2, 2); }

inline Eigen::MatrixXcd make_X() { return make_2x2_matrix(0, 1, 1, 0); }

inline Eigen::MatrixXcd make_Y() { return make_2x2_matrix(0, -1.i, 1.i, 0); }

inline Eigen::MatrixXcd make_Z() { return make_2x2_matrix(1, 0, 0, -1); }

inline Eigen::MatrixXcd make_H() {
    return make_2x2_matrix(1 / sqrt(2.), 1 / sqrt(2.), 1 / sqrt(2.), -1 / sqrt(2.));
}

inline Eigen::MatrixXcd make_S() { return make_2x2_matrix(1, 0, 0, 1.i); }

inline Eigen::MatrixXcd make_Sdag() { return make_2x2_matrix(1, 0, 0, -1.i); }

inline Eigen::MatrixXcd make_T() { return make_2x2_matrix(1, 0, 0, (1. + 1.i) / sqrt(2.)); }

inline Eigen::MatrixXcd make_Tdag() { return make_2x2_matrix(1, 0, 0, (1. - 1.i) / sqrt(2.)); }

inline Eigen::MatrixXcd make_sqrtX() {
    return make_2x2_matrix(0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i);
}

inline Eigen::MatrixXcd make_sqrtY() {
    return make_2x2_matrix(0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i);
}

inline Eigen::MatrixXcd make_sqrtXdag() {
    return make_2x2_matrix(0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i);
}

inline Eigen::MatrixXcd make_sqrtYdag() {
    return make_2x2_matrix(0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i);
}

inline Eigen::MatrixXcd make_P0() { return make_2x2_matrix(1, 0, 0, 0); }

inline Eigen::MatrixXcd make_P1() { return make_2x2_matrix(0, 0, 0, 1); }

inline Eigen::MatrixXcd make_RX(double angle) {
    return make_2x2_matrix(std::cos(angle / 2),
                           -1i * std::sin(angle / 2),
                           -1i * std::sin(angle / 2),
                           std::cos(angle / 2));
}

inline Eigen::MatrixXcd make_RY(double angle) {
    return make_2x2_matrix(
        std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2));
}

inline Eigen::MatrixXcd make_RZ(double angle) {
    return make_2x2_matrix(std::exp(-1i * (angle / 2)), 0, 0, std::exp(1i * (angle / 2)));
}

inline Eigen::MatrixXcd make_U(double theta, double phi, double lambda) {
    return make_2x2_matrix(std::cos(theta / 2.),
                           -std::exp(1i * lambda) * std::sin(theta / 2.),
                           std::exp(1i * phi) * std::sin(theta / 2.),
                           std::exp(1i * phi) * std::exp(1i * lambda) * std::cos(theta / 2.));
}
