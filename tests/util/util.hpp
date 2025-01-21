#pragma once

#include <Eigen/Core>
#include <complex>
#include <scaluq/types.hpp>

using namespace scaluq;
using namespace std::complex_literals;

using scaluq::internal::ComplexMatrix;

template <std::floating_point Fp>
using ComplexVector = Eigen::Matrix<StdComplex<Fp>, -1, 1>;

template <std::floating_point Fp>
constexpr Fp eps_() {
    if constexpr (std::is_same_v<Fp, double>)
        return 1e-12;
    else if constexpr (std::is_same_v<Fp, float>)
        return 1e-4;
    else
        static_assert(internal::lazy_false_v<Fp>, "unknown GateImpl");
}
template <std::floating_point Fp>
constexpr Fp eps = eps_<Fp>();

template <std::floating_point Fp>
inline void check_near(const StdComplex<Fp>& a, const StdComplex<Fp>& b) {
    ASSERT_LE(std::abs(a - b), eps<Fp>);
}

template <std::floating_point Fp, ExecutionSpace Sp>
inline bool same_state(const StateVector<Fp, Sp>& s1,
                       const StateVector<Fp, Sp>& s2,
                       const Fp e = eps<Fp>) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs((std::complex<Fp>)s1_cp[i] - (std::complex<Fp>)s2_cp[i]) > e) return false;
    }
    return true;
};

template <std::floating_point Fp, ExecutionSpace Sp>
inline bool same_state_except_global_phase(const StateVector<Fp, Sp>& s1,
                                           const StateVector<Fp, Sp>& s2,
                                           const Fp e = eps<Fp>) {
    auto s1_cp = s1.get_amplitudes();
    auto s2_cp = s2.get_amplitudes();
    assert(s1.n_qubits() == s2.n_qubits());
    std::uint64_t significant = 0;
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs((std::complex<Fp>)s1_cp[i]) > std::abs((std::complex<Fp>)s1_cp[significant])) {
            significant = i;
        }
    }
    if (std::abs((std::complex<Fp>)s1_cp[significant]) < e) {
        for (std::uint64_t i = 0; i < s2.dim(); ++i) {
            if (std::abs((std::complex<Fp>)s2_cp[i]) > e) return false;
        }
        return true;
    }
    Fp phase = std::arg(std::complex<Fp>(s2_cp[significant] / s1_cp[significant]));
    std::complex<Fp> phase_coef = std::polar(1., phase);
    for (std::uint64_t i = 0; i < s1.dim(); ++i) {
        if (std::abs(phase_coef * (std::complex<Fp>)s1_cp[i] - (std::complex<Fp>)s2_cp[i]) > e)
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
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_single_Pauli(std::uint64_t pauli_id) {
    ComplexMatrix<Fp> mat(2, 2);
    if (pauli_id == 0)
        mat << 1, 0, 0, 1;
    else if (pauli_id == 1)
        mat << 0, 1, 1, 0;
    else if (pauli_id == 2)
        mat << 0, Complex<Fp>(0, -1), Complex<Fp>(0, 1), 0;
    else if (pauli_id == 3)
        mat << 1, 0, 0, -1;
    return mat;
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_random_one_target_unitary() {
    ComplexMatrix<Fp> Identity, X, Y, Z;
    Identity = get_eigen_matrix_single_Pauli<Fp>(0);
    X = get_eigen_matrix_single_Pauli<Fp>(1);
    Y = get_eigen_matrix_single_Pauli<Fp>(2);
    Z = get_eigen_matrix_single_Pauli<Fp>(3);

    Fp icoef, xcoef, ycoef, zcoef, norm;
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
    return icoef * Identity + Complex<Fp>(0, 1) * xcoef * X + Complex<Fp>(0, 1) * ycoef * Y +
           Complex<Fp>(0, 1) * zcoef * Z;
}
template <std::floating_point Fp>
inline ComplexVector<Fp> get_eigen_diagonal_matrix_random_multi_qubit_unitary(
    std::uint64_t qubit_count) {
    std::uint64_t dim = (1ULL) << qubit_count;
    auto vec = ComplexVector<Fp>(dim);
    Random random;
    for (std::uint64_t i = 0; i < dim; ++i) {
        Fp angle = random.uniform() * 2 * 3.14159;
        vec[i] = std::cos(angle) + Complex<Fp>(0, 1) * std::sin(angle);
    }
    return vec;
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_expanded_eigen_matrix_with_identity(
    std::uint64_t target_qubit_index,
    const ComplexMatrix<Fp>& one_target_matrix,
    std::uint64_t qubit_count) {
    const std::uint64_t left_dim = 1ULL << target_qubit_index;
    const std::uint64_t right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = ComplexMatrix<Fp>::Identity(left_dim, left_dim);
    auto right_identity = ComplexMatrix<Fp>::Identity(right_dim, right_dim);
    return internal::kronecker_product<Fp>(
        internal::kronecker_product<Fp>(right_identity, one_target_matrix), left_identity);
}

// get expanded matrix
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> pauli_ids) {
    ComplexMatrix<Fp> result = ComplexMatrix<Fp>::Identity(1, 1);
    for (std::uint64_t i = 0; i < pauli_ids.size(); ++i) {
        result =
            internal::kronecker_product<Fp>(get_eigen_matrix_single_Pauli<Fp>(pauli_ids[i]), result)
                .eval();
    }
    return result;
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_full_qubit_pauli(std::vector<std::uint64_t> index_list,
                                                           std::vector<std::uint64_t> pauli_list,
                                                           std::uint64_t qubit_count) {
    std::vector<std::uint64_t> whole_pauli_ids(qubit_count, 0);
    for (std::uint64_t i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli<Fp>(whole_pauli_ids);
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_full_qubit_CX(std::uint64_t control_qubit_index,
                                                        std::uint64_t target_qubit_index,
                                                        std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix<Fp> result = ComplexMatrix<Fp>::Zero(dim, dim);
    for (std::uint64_t ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_full_qubit_CZ(std::uint64_t control_qubit_index,
                                                        std::uint64_t target_qubit_index,
                                                        std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix<Fp> result = ComplexMatrix<Fp>::Zero(dim, dim);
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
template <std::floating_point Fp>
inline ComplexMatrix<Fp> get_eigen_matrix_full_qubit_Swap(std::uint64_t target_qubit_index1,
                                                          std::uint64_t target_qubit_index2,
                                                          std::uint64_t qubit_count) {
    std::uint64_t dim = 1ULL << qubit_count;
    ComplexMatrix<Fp> result = ComplexMatrix<Fp>::Zero(dim, dim);
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

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_2x2_matrix(const StdComplex<Fp>& a00,
                                         const StdComplex<Fp>& a01,
                                         const StdComplex<Fp>& a10,
                                         const StdComplex<Fp>& a11) {
    ComplexMatrix<Fp> m(2, 2);
    m << a00, a01, a10, a11;
    return m;
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_I() {
    return ComplexMatrix<Fp>::Identity(2, 2);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_X() {
    return make_2x2_matrix<Fp>(0, 1, 1, 0);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_Y() {
    return make_2x2_matrix<Fp>(0, StdComplex<Fp>(0, -1), StdComplex<Fp>(0, 1), 0);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_Z() {
    return make_2x2_matrix<Fp>(1, 0, 0, -1);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_H() {
    return make_2x2_matrix<Fp>(1 / sqrt(2), 1 / sqrt(2), 1 / sqrt(2), -1 / sqrt(2));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_S() {
    return make_2x2_matrix<Fp>(1, 0, 0, StdComplex<Fp>(0, 1));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_T() {
    return make_2x2_matrix<Fp>(1, 0, 0, StdComplex<Fp>(1, 1) / (Fp)sqrt(2.));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_Sdag() {
    return make_2x2_matrix<Fp>(1, 0, 0, Complex<Fp>(0, -1));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_Tdag() {
    return make_2x2_matrix<Fp>(1, 0, 0, StdComplex<Fp>(1, -1) / (Fp)sqrt(2.));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_SqrtX() {
    return make_2x2_matrix<Fp>(StdComplex<Fp>(0.5, 0.5),
                               StdComplex<Fp>(0.5, -0.5),
                               StdComplex<Fp>(0.5, -0.5),
                               StdComplex<Fp>(0.5, 0.5));
}
template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_SqrtY() {
    return make_2x2_matrix<Fp>(StdComplex<Fp>(0.5, 0.5),
                               StdComplex<Fp>(-0.5, -0.5),
                               StdComplex<Fp>(0.5, 0.5),
                               StdComplex<Fp>(0.5, 0.5));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_SqrtXdag() {
    return make_2x2_matrix<Fp>(StdComplex<Fp>(0.5, -0.5),
                               StdComplex<Fp>(0.5, 0.5),
                               StdComplex<Fp>(0.5, 0.5),
                               StdComplex<Fp>(0.5, -0.5));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_SqrtYdag() {
    return make_2x2_matrix<Fp>(StdComplex<Fp>(0.5, -0.5),
                               StdComplex<Fp>(0.5, -0.5),
                               StdComplex<Fp>(-0.5, 0.5),
                               StdComplex<Fp>(0.5, -0.5));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_P0() {
    return make_2x2_matrix<Fp>(1, 0, 0, 0);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_P1() {
    return make_2x2_matrix<Fp>(0, 0, 0, 1);
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_RX(Fp angle) {
    return make_2x2_matrix<Fp>(std::cos(angle / 2),
                               StdComplex<Fp>(0, -std::sin(angle / 2)),
                               StdComplex<Fp>(0, -std::sin(angle / 2)),
                               std::cos(angle / 2));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_RY(Fp angle) {
    return make_2x2_matrix<Fp>(
        std::cos(angle / 2), -std::sin(angle / 2), std::sin(angle / 2), std::cos(angle / 2));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_RZ(Fp angle) {
    return make_2x2_matrix<Fp>(
        std::exp(StdComplex<Fp>(0, -angle / 2)), 0, 0, std::exp(StdComplex<Fp>(0, angle / 2)));
}

template <std::floating_point Fp>
inline ComplexMatrix<Fp> make_U(Fp theta, Fp phi, Fp lambda) {
    return make_2x2_matrix<Fp>(std::cos(theta / 2.),
                               -std::exp(StdComplex<Fp>(0, lambda)) * std::sin(theta / 2),
                               std::exp(StdComplex<Fp>(0, phi)) * std::sin(theta / 2),
                               std::exp(StdComplex<Fp>(0, phi)) *
                                   std::exp(StdComplex<Fp>(0, lambda)) * std::cos(theta / 2));
}
