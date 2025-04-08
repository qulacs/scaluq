#pragma once

#include <Eigen/Core>
#include <complex>
#include <scaluq/types.hpp>

using namespace scaluq;
using namespace std::complex_literals;

using ComplexVector = Eigen::Matrix<StdComplex, -1, 1>;

template <Precision Prec>
constexpr double eps_() {
    if constexpr (Prec == Precision::F16)
        return 1e-2;
    else if constexpr (Prec == Precision::F32)
        return 1e-4;
    else if constexpr (Prec == Precision::F64)
        return 1e-12;
    else if constexpr (Prec == Precision::BF16)
        return 1e-1;
    else
        static_assert(internal::lazy_false_v<internal::Float<Prec>>, "unknown Precision");
}
template <Precision Prec>
constexpr double eps = eps_<Prec>();

template <Precision Prec>
inline void check_near(const StdComplex& a, const StdComplex& b) {
    ASSERT_LE(std::abs(a - b), eps<Prec>);
}

template <Precision Prec, ExecutionSpace Space>
inline bool same_state(const StateVector<Prec, Space>& s1,
                       const StateVector<Prec, Space>& s2,
                       const double e = eps<Prec>) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs(s1_cp[i] - s2_cp[i]) > e) return false;
    }
    return true;
};

template <Precision Prec, ExecutionSpace Space>
inline bool same_state_except_global_phase(const StateVector<Prec, Space>& s1,
                                           const StateVector<Prec, Space>& s2,
                                           const double e = eps<Prec>) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    std::uint64_t significant = 0;
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs(s1_cp[i]) > std::abs(s1_cp[significant])) {
            significant = i;
        }
    }
    if (std::abs(s1_cp[significant]) < e) {
        for (std::uint64_t i = 0; i < s2.dim(); ++i) {
            if (std::abs(s2_cp[i]) > e) return false;
        }
        return true;
    }
    double phase = std::arg(s2_cp[significant] / s1_cp[significant]);
    StdComplex phase_coef = std::polar(1., phase);
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs(phase_coef * s1_cp[i] - s2_cp[i]) > e) return false;
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
inline ComplexMatrix get_eigen_matrix_single_Pauli(std::uint64_t pauli_id) {
    ComplexMatrix mat(2, 2);
    if (pauli_id == 0)
        mat << 1, 0, 0, 1;
    else if (pauli_id == 1)
        mat << 0, 1, 1, 0;
    else if (pauli_id == 2)
        mat << 0, StdComplex(0, -1), StdComplex(0, 1), 0;
    else if (pauli_id == 3)
        mat << 1, 0, 0, -1;
    return mat;
}
inline ComplexMatrix get_eigen_matrix_random_one_target_unitary() {
    ComplexMatrix Identity, X, Y, Z;
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
    return icoef * Identity + StdComplex(0, 1) * xcoef * X + StdComplex(0, 1) * ycoef * Y +
           StdComplex(0, 1) * zcoef * Z;
}
inline ComplexVector get_eigen_diagonal_matrix_random_multi_qubit_unitary(
    std::uint64_t qubit_count) {
    std::uint64_t dim = (1ULL) << qubit_count;
    auto vec = ComplexVector(dim);
    Random random;
    for (std::uint64_t i = 0; i < dim; ++i) {
        double angle = random.uniform() * 2 * std::numbers::pi;
        vec[i] = std::cos(angle) + StdComplex(0, 1) * std::sin(angle);
    }
    return vec;
}
inline ComplexMatrix get_expanded_eigen_matrix_with_identity(std::uint64_t target_qubit_index,
                                                             const ComplexMatrix& one_target_matrix,
                                                             std::uint64_t qubit_count) {
    const std::uint64_t left_dim = 1ULL << target_qubit_index;
    const std::uint64_t right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = ComplexMatrix::Identity(left_dim, left_dim);
    auto right_identity = ComplexMatrix::Identity(right_dim, right_dim);
    return internal::kronecker_product(
        internal::kronecker_product(right_identity, one_target_matrix), left_identity);
}

// get expanded matrix
inline ComplexMatrix get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> pauli_ids) {
    ComplexMatrix result = ComplexMatrix::Identity(1, 1);
    for (std::uint64_t i = 0; i < pauli_ids.size(); ++i) {
        result =
            internal::kronecker_product(get_eigen_matrix_single_Pauli(pauli_ids[i]), result).eval();
    }
    return result;
}
inline ComplexMatrix get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> index_list,
                                                       std::vector<std::uint64_t> pauli_list,
                                                       std::uint64_t qubit_count) {
    std::vector<std::uint64_t> whole_pauli_ids(qubit_count, 0);
    for (std::uint64_t i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli(whole_pauli_ids);
}
inline ComplexMatrix get_eigen_matrix_full_qubit_CX(std::uint64_t control_qubit_index,
                                                    std::uint64_t target_qubit_index,
                                                    std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix result = ComplexMatrix::Zero(dim, dim);
    for (std::uint64_t ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
inline ComplexMatrix get_eigen_matrix_full_qubit_CZ(std::uint64_t control_qubit_index,
                                                    std::uint64_t target_qubit_index,
                                                    std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix result = ComplexMatrix::Zero(dim, dim);
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
inline ComplexMatrix get_eigen_matrix_full_qubit_Swap(std::uint64_t target_qubit_index1,
                                                      std::uint64_t target_qubit_index2,
                                                      std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix result = ComplexMatrix::Zero(dim, dim);
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

inline ComplexMatrix make_2x2_matrix(const StdComplex& a00,
                                     const StdComplex& a01,
                                     const StdComplex& a10,
                                     const StdComplex& a11) {
    ComplexMatrix m(2, 2);
    m << a00, a01, a10, a11;
    return m;
}

inline ComplexMatrix make_I() { return ComplexMatrix::Identity(2, 2); }

inline ComplexMatrix make_X() { return make_2x2_matrix(0, 1, 1, 0); }

inline ComplexMatrix make_Y() { return make_2x2_matrix(0, StdComplex(0, -1), StdComplex(0, 1), 0); }

inline ComplexMatrix make_Z() { return make_2x2_matrix(1, 0, 0, -1); }

inline ComplexMatrix make_H() {
    return make_2x2_matrix(1 / std::sqrt(2), 1 / std::sqrt(2), 1 / std::sqrt(2), -1 / std::sqrt(2));
}
inline ComplexMatrix make_S() { return make_2x2_matrix(1, 0, 0, StdComplex(0, 1)); }
inline ComplexMatrix make_T() { return make_2x2_matrix(1, 0, 0, StdComplex(1, 1) / std::sqrt(2.)); }
inline ComplexMatrix make_Sdag() { return make_2x2_matrix(1, 0, 0, StdComplex(0, -1)); }
inline ComplexMatrix make_Tdag() {
    return make_2x2_matrix(1, 0, 0, StdComplex(1, -1) / std::sqrt(2.));
}
inline ComplexMatrix make_SqrtX() {
    return make_2x2_matrix(
        StdComplex(0.5, 0.5), StdComplex(0.5, -0.5), StdComplex(0.5, -0.5), StdComplex(0.5, 0.5));
}
inline ComplexMatrix make_SqrtY() {
    return make_2x2_matrix(
        StdComplex(0.5, 0.5), StdComplex(-0.5, -0.5), StdComplex(0.5, 0.5), StdComplex(0.5, 0.5));
}

inline ComplexMatrix make_SqrtXdag() {
    return make_2x2_matrix(
        StdComplex(0.5, -0.5), StdComplex(0.5, 0.5), StdComplex(0.5, 0.5), StdComplex(0.5, -0.5));
}

inline ComplexMatrix make_SqrtYdag() {
    return make_2x2_matrix(
        StdComplex(0.5, -0.5), StdComplex(0.5, -0.5), StdComplex(-0.5, 0.5), StdComplex(0.5, -0.5));
}

inline ComplexMatrix make_P0() { return make_2x2_matrix(1, 0, 0, 0); }

inline ComplexMatrix make_P1() { return make_2x2_matrix(0, 0, 0, 1); }

inline ComplexMatrix make_RX(double angle) {
    return make_2x2_matrix(std::cos(angle / 2),
                           StdComplex(0, -std::sin(angle / 2)),
                           StdComplex(0, -std::sin(angle / 2)),
                           std::cos(angle / 2));
}

inline ComplexMatrix make_RY(double angle) {
    return make_2x2_matrix(
        std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2));
}

inline ComplexMatrix make_RZ(double angle) {
    return make_2x2_matrix(
        std::exp(StdComplex(0, -angle / 2)), 0, 0, std::exp(StdComplex(0, angle / 2)));
}

inline ComplexMatrix make_U(double theta, double phi, double lambda) {
    return make_2x2_matrix(
        std::cos(theta / 2.),
        -std::exp(StdComplex(0, lambda)) * std::sin(theta / 2),
        std::exp(StdComplex(0, phi)) * std::sin(theta / 2),
        std::exp(StdComplex(0, phi)) * std::exp(StdComplex(0, lambda)) * std::cos(theta / 2));
}
