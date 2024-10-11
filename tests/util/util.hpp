#pragma once

#include <Eigen/Core>
#include <complex>
using namespace std::complex_literals;

#include <types.hpp>
#include <util/random.hpp>
#include <util/utility.hpp>
using namespace scaluq;

template <std::floating_point FloatType>
inline bool same_state(const StateVector<FloatType>& s1,
                       const StateVector<FloatType>& s2,
                       const double eps = 1e-12) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs((std::complex<double>)s1_cp[i] - (std::complex<double>)s2_cp[i]) > eps)
            return false;
    }
    return true;
};

inline bool same_state_except_global_phase(const StateVector<double>& s1,
                                           const StateVector<double>& s2,
                                           const double eps = 1e-12) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    std::uint64_t significant = 0;
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs((std::complex<double>)s1_cp[i]) >
            std::abs((std::complex<double>)s1_cp[significant])) {
            significant = i;
        }
    }
    if (std::abs((std::complex<double>)s1_cp[significant]) < eps) {
        for (std::uint64_t i = 0; i < s2.dim(); ++i) {
            if (std::abs((std::complex<double>)s2_cp[i]) > eps) return false;
        }
        return true;
    }
    double phase = std::arg(std::complex<double>(s2_cp[significant] / s1_cp[significant]));
    std::complex<double> phase_coef = std::polar(1., phase);
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs(phase_coef * (std::complex<double>)s1_cp[i] - (std::complex<double>)s2_cp[i]) >
            eps)
            return false;
    }
    return true;
};

#define _CHECK_GT(val1, val2) _check_gt(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
inline std::string _check_gt(T val1,
                             T val2,
                             std::string val1_name,
                             std::string val2_name,
                             std::string file,
                             std::uint64_t line) {
    if (val1 > val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") > (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

// obtain single dense matrix
inline Eigen::MatrixXcd get_eigen_matrix_single_Pauli(std::uint64_t pauli_id) {
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
inline Eigen::MatrixXcd get_eigen_matrix_random_one_target_unitary() {
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
inline Eigen::VectorXcd get_eigen_diagonal_matrix_random_multi_qubit_unitary(
    std::uint64_t qubit_count) {
    std::uint64_t dim = (1ULL) << qubit_count;
    auto vec = Eigen::VectorXcd(dim);
    Random random;
    for (std::uint64_t i = 0; i < dim; ++i) {
        double angle = random.uniform() * 2 * 3.14159;
        vec[i] = std::cos(angle) + 1.i * std::sin(angle);
    }
    return vec;
}

inline Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(
    std::uint64_t target_qubit_index,
    const Eigen::MatrixXcd& one_target_matrix,
    std::uint64_t qubit_count) {
    const std::uint64_t left_dim = 1ULL << target_qubit_index;
    const std::uint64_t right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return internal::kronecker_product(
        internal::kronecker_product(right_identity, one_target_matrix), left_identity);
}

// get expanded matrix
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> pauli_ids) {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(1, 1);
    for (std::uint64_t i = 0; i < pauli_ids.size(); ++i) {
        result =
            internal::kronecker_product(get_eigen_matrix_single_Pauli(pauli_ids[i]), result).eval();
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> index_list,
                                                          std::vector<std::uint64_t> pauli_list,
                                                          std::uint64_t qubit_count) {
    std::vector<std::uint64_t> whole_pauli_ids(qubit_count, 0);
    for (std::uint64_t i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli(whole_pauli_ids);
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_CX(std::uint64_t control_qubit_index,
                                                       std::uint64_t target_qubit_index,
                                                       std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_CZ(std::uint64_t control_qubit_index,
                                                       std::uint64_t target_qubit_index,
                                                       std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t ind = 0; ind < dim; ++ind) {
        if ((ind & (1ULL << control_qubit_index)) != 0 &&
            (ind & (1ULL << target_qubit_index)) != 0) {
            result(ind, ind) = -1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
inline Eigen::MatrixXcd get_eigen_matrix_full_qubit_Swap(std::uint64_t target_qubit_index1,
                                                         std::uint64_t target_qubit_index2,
                                                         std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t ind = 0; ind < dim; ++ind) {
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
inline Eigen::MatrixXcd make_T() { return make_2x2_matrix(1, 0, 0, (1. + 1.i) / sqrt(2.)); }
inline Eigen::MatrixXcd make_Sdag() { return make_2x2_matrix(1, 0, 0, -1.i); }
inline Eigen::MatrixXcd make_Tdag() { return make_2x2_matrix(1, 0, 0, (1. - 1.i) / sqrt(2.)); }
inline Eigen::MatrixXcd make_SqrtX() {
    return make_2x2_matrix(0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i);
}
inline Eigen::MatrixXcd make_SqrtY() {
    return make_2x2_matrix(0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i);
}

inline Eigen::MatrixXcd make_SqrtXdag() {
    return make_2x2_matrix(0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i);
}

inline Eigen::MatrixXcd make_SqrtYdag() {
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
