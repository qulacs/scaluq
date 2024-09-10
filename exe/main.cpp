#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"
#include "../scaluq/util/utility.hpp"

using namespace scaluq;
using namespace std;

template <bool enable_validate = true>
inline std::uint64_t vector_to_mask(const std::vector<std::uint64_t>& v) {
    std::uint64_t mask = 0;
    for (auto x : v) {
        if constexpr (enable_validate) {
            if (x >= sizeof(std::uint64_t) * 8) [[unlikely]] {
                throw std::runtime_error("The size of the qubit system must be less than 64.");
            }
            if ((mask >> x) & 1) [[unlikely]] {
                throw std::runtime_error("The specified qubit is duplicated.");
            }
        }
        mask |= 1ULL << x;
    }
    return mask;
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

inline std::vector<std::uint64_t> create_matrix_mask_list(std::uint64_t target_mask) {
    std::vector<std::uint64_t> bit_mask_list;
    std::uint64_t x = 1;
    for (std::uint64_t bit = 0; bit < 64; bit++) {
        if (target_mask & x) bit_mask_list.emplace_back(x);
        x <<= 1;
    }
    const std::uint64_t qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << qubit_index_count;
    std::vector<std::uint64_t> mask_list(matrix_dim, 0);

    for (std::uint64_t cursor = 0; cursor < matrix_dim; cursor++) {
        for (std::uint64_t idx = 0; idx < bit_mask_list.size(); idx++) {
            std::uint64_t bit_mask = bit_mask_list[idx];
            if (cursor & bit_mask) mask_list[cursor] ^= bit_mask;
        }
    }
    return mask_list;
}

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
inline Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
inline std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        host_vector.data(), host_vector.size());
    Kokkos::deep_copy(host_view, device_view);
    return host_vector;
}

KOKKOS_INLINE_FUNCTION std::uint64_t insert_zero_at_mask_positions(std::uint64_t basis_index,
                                                                   std::uint64_t insert_mask) {
    for (std::uint64_t bit_mask = insert_mask; bit_mask;
         bit_mask &= (bit_mask - 1)) {  // loop through set bits
        std::uint64_t lower_mask = ~bit_mask & (bit_mask - 1);
        std::uint64_t upper_mask = ~lower_mask;
        basis_index = ((basis_index & upper_mask) << 1) | (basis_index & lower_mask);
    }
    return basis_index;
}

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

void single_target_dense_matrix_gate_view(std::uint64_t target_mask,
                                          const Matrix& matrix,
                                          StateVector& state) {
    const std::uint64_t loop_dim = state.dim() >> 1;
    const std::uint64_t mask_low = target_mask - 1;
    const std::uint64_t mask_high = ~mask_low;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const std::uint64_t state_index) {
            std::uint64_t basis_0 = (state_index & mask_low) + ((state_index & mask_high) << 1);
            std::uint64_t basis_1 = basis_0 + target_mask;
            Complex v0 = state._raw[basis_0];
            Complex v1 = state._raw[basis_1];
            state._raw[basis_0] = matrix(0, 0) * v0 + matrix(0, 1) * v1;
            state._raw[basis_1] = matrix(1, 0) * v0 + matrix(1, 1) * v1;
        });
    Kokkos::fence();
}

void double_target_dense_matrix_gate(std::uint64_t target_mask,
                                     const Matrix& matrix,
                                     StateVector& state) {
    std::uint64_t lower_target_mask = -target_mask & target_mask;
    std::uint64_t upper_target_mask = target_mask ^ lower_target_mask;

    const std::uint64_t loop_dim = state.dim() >> 2;

    Kokkos::parallel_for(
        loop_dim, KOKKOS_LAMBDA(const std::uint64_t it) {
            std::uint64_t basis_0 = insert_zero_at_mask_positions(it, target_mask);
            std::uint64_t basis_1 = basis_0 | lower_target_mask;
            std::uint64_t basis_2 = basis_0 | upper_target_mask;
            std::uint64_t basis_3 = basis_1 | target_mask;
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix(0, 0) * val0 + matrix(0, 1) * val1 + matrix(0, 2) * val2 +
                           matrix(0, 3) * val3;
            Complex res1 = matrix(1, 0) * val0 + matrix(1, 1) * val1 + matrix(1, 2) * val2 +
                           matrix(1, 3) * val3;
            Complex res2 = matrix(2, 0) * val0 + matrix(2, 1) * val1 + matrix(2, 2) * val2 +
                           matrix(2, 3) * val3;
            Complex res3 = matrix(3, 0) * val0 + matrix(3, 1) * val1 + matrix(3, 2) * val2 +
                           matrix(3, 3) * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
    Kokkos::fence();
}

void multi_target_dense_matrix_gate_parallel(std::uint64_t target_mask,
                                             const Matrix& matrix,
                                             StateVector& state) {
    const std::uint64_t target_qubit_index_count = std::popcount(target_mask);
    const std::uint64_t matrix_dim = 1ULL << target_qubit_index_count;
    const std::uint64_t loop_dim = state.dim() >> target_qubit_index_count;
    std::vector<std::uint64_t> matrix_mask_list = create_matrix_mask_list(target_mask);
    Kokkos::View<std::uint64_t*> matrix_mask_view =
        convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            Kokkos::View<Complex*> buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            std::uint64_t basis_0 = team.league_rank();
            basis_0 = insert_zero_at_mask_positions(basis_0, target_mask);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](std::uint64_t y) {
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](std::uint64_t x, Complex& inner_sum) {
                        inner_sum += matrix(y, x) * state._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim),
                                 [=](const std::uint64_t y) {
                                     state._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
                                 });
        });
    Kokkos::fence();
}

void test_dense(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 10;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> U1, U2, U3;
    std::vector<std::uint64_t> targets(3);
    std::vector<std::uint64_t> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    int fail_count_single = 0;
    int fail_count_double = 0;
    int fail_count_triple = 0;
    for (std::uint64_t i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    // general single
    {
        Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            ComplexMatrix mat(U1.rows(), U1.cols());
            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            Umerge = U1;
            test_state =
                get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) * test_state;
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    mat(i, j) = U1(i, j);
                }
            }
            std::vector<std::uint64_t> target_list = {targets[0]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, mat, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                if (std::abs((StdComplex)state_cp[i] - test_state[i]) > 1e-10) {
                    std::cout << "i: " << i << " state_cp[i]: " << state_cp[i]
                              << " test_state[i]: " << test_state[i] << std::endl;
                    fail_count_single++;
                }
            }
        }
    }
    // general double
    {
        Eigen::Matrix<StdComplex, 4, 4, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            U2 = get_eigen_matrix_random_one_target_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            if (targets[0] > targets[1]) {
                std::swap(targets[0], targets[1]);
            }
            Umerge = internal::kronecker_product(U2, U1);
            ComplexMatrix mat(Umerge.rows(), Umerge.cols());
            test_state = get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    mat(i, j) = Umerge(i, j);
                }
            }
            std::vector<std::uint64_t> target_list = {targets[0], targets[1]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, mat, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                if (std::abs((StdComplex)state_cp[i] - test_state[i]) > 1e-10) {
                    std::cout << "i: " << i << " state_cp[i]: " << state_cp[i]
                              << " test_state[i]: " << test_state[i] << std::endl;
                    fail_count_double++;
                }
            }
        }
    }
    // general triple
    {
        Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
        for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
            StateVector state = StateVector::Haar_random_state(n_qubits);
            auto state_cp = state.amplitudes();
            for (std::uint64_t i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }
            U1 = get_eigen_matrix_random_one_target_unitary();
            U2 = get_eigen_matrix_random_one_target_unitary();
            U3 = get_eigen_matrix_random_one_target_unitary();

            std::shuffle(index_list.begin(), index_list.end(), engine);
            targets[0] = index_list[0];
            targets[1] = index_list[1];
            targets[2] = index_list[2];
            // std::sort(targets.begin(), targets.end());
            Umerge = internal::kronecker_product(U3, internal::kronecker_product(U2, U1));
            ComplexMatrix mat(Umerge.rows(), Umerge.cols());

            test_state = get_expanded_eigen_matrix_with_identity(targets[2], U3, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[1], U2, n_qubits) *
                         get_expanded_eigen_matrix_with_identity(targets[0], U1, n_qubits) *
                         test_state;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    mat(i, j) = Umerge(i, j);
                }
            }
            std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
            std::vector<std::uint64_t> control_list = {};
            Gate dense_gate = gate::DenseMatrix(target_list, mat, control_list);
            dense_gate->update_quantum_state(state);
            state_cp = state.amplitudes();
            int flag = 0;
            for (std::uint64_t i = 0; i < dim; i++) {
                if (std::abs((StdComplex)state_cp[i] - test_state[i]) > 1e-10) {
                    // std::cout << "i: " << i << " state_cp[i]: " << state_cp[i]
                    //           << " test_state[i]: " << test_state[i] << std::endl;
                    fail_count_triple++;
                    flag = 1;
                }
            }
            if (flag) {
                flag = 0;
                std::cout << target_list[0] << " " << target_list[1] << " " << target_list[2]
                          << std::endl;
            }
        }
    }
    // std::cout << "fail_count_single: " << fail_count_single << std::endl;
    // std::cout << "fail_count_double: " << fail_count_double << std::endl;
    // std::cout << "fail_count_triple: " << fail_count_triple << std::endl;
}

void run() {
    std::uint64_t n_qubits = 4;
    test_dense(n_qubits);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
