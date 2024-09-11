#include <bitset>  // 後で消す
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"
#include "../tests/util/util.hpp"

using namespace scaluq;
using namespace std;

double eps = 1e-12;

#define ASSERT_NEAR(a, b, c)              \
    if (abs(a - b) >= c) {                \
        std::cout << "fail" << std::endl; \
        return;                           \
    }

template <std::uint64_t num_target>
void test_target_control(std::uint64_t n_qubits) {
    Random random;
    std::vector<std::uint64_t> shuffled(n_qubits);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n_qubits) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::vector<std::uint64_t> targets(num_target);
    for (std::uint64_t i : std::views::iota(0ULL, num_target)) {
        targets[i] = shuffled[i];
    }
    std::uint64_t num_control = random.int32() % (n_qubits - num_target + 1);
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[num_target + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    for (auto i : targets) {
        if (i >= n_qubits) {
            cout << "Error: target qubit index is out of range" << endl;
            return;
        }
    }
    for (auto i : controls) {
        if (i >= n_qubits) {
            cout << "Error: control qubit index is out of range" << endl;
            return;
        }
    }
}

void test_general_matrix_internal(Gate gate_control,
                                  Gate gate_simple,
                                  std::uint64_t n_qubits,
                                  std::uint64_t control_mask) {
    StateVector state = StateVector::Haar_random_state(n_qubits);
    std::uint64_t dim = state.dim();
    auto amplitudes = state.get_amplitudes();
    StateVector state_controlled(n_qubits);
    std::vector<Complex> amplitudes_controlled(dim);
    for (std::uint64_t i : std::views::iota(0ULL, dim >> std::popcount(control_mask))) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) | control_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state);
    gate_simple->update_quantum_state(state_controlled);
    amplitudes = state.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        ASSERT_NEAR(
            Kokkos::abs(amplitudes_controlled[i] -
                        amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                   control_mask]),
            0.,
            eps);
    }
}

template <std::uint64_t num_target>
void test_general_matrix(std::uint64_t n_qubits) {
    Random random;
    std::vector<std::uint64_t> shuffled(n_qubits);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n_qubits) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::vector<std::uint64_t> targets(num_target);
    for (std::uint64_t i : std::views::iota(0ULL, num_target)) {
        targets[i] = shuffled[i];
    }
    std::uint64_t num_control = random.int32() % (n_qubits - num_target + 1);
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[num_target + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    if constexpr (num_target == 1) {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U =
            get_eigen_matrix_random_one_target_unitary();
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(targets, mat.sparseView(), {});
        test_general_matrix_internal(d1, d2, n_qubits, control_mask);
        test_general_matrix_internal(s1, s2, n_qubits, control_mask);
    } else if constexpr (num_target == 2) {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary();
        auto U = internal::kronecker_product(U2, U1);
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(targets, mat.sparseView(), {});
        test_general_matrix_internal(d1, d2, n_qubits, control_mask);
        test_general_matrix_internal(s1, s2, n_qubits, control_mask);
    } else {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U2 =
            get_eigen_matrix_random_one_target_unitary();
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U3 =
            get_eigen_matrix_random_one_target_unitary();
        auto U = internal::kronecker_product(U3, internal::kronecker_product(U2, U1));
        internal::ComplexMatrix mat(U.rows(), U.cols());
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                mat(i, j) = U(i, j);
            }
        }
        Gate d1 = gate::DenseMatrix(targets, mat, controls);
        Gate d2 = gate::DenseMatrix(targets, mat, {});
        Gate s1 = gate::SparseMatrix(targets, mat.sparseView(), controls);
        Gate s2 = gate::SparseMatrix(targets, mat.sparseView(), {});
        test_general_matrix_internal(d1, d2, n_qubits, control_mask);
        test_general_matrix_internal(s1, s2, n_qubits, control_mask);
    }
}

void run() {
    test_general_matrix<1>(10);
    test_general_matrix<2>(10);
    test_general_matrix<3>(10);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
