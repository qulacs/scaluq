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

inline SparseComplexMatrix make_sparse_complex_matrix(const std::uint64_t n_qubit,
                                                      double border = 0.9) {
    const std::uint64_t dim = 1ULL << n_qubit;
    typedef Eigen::Triplet<StdComplex> T;
    std::vector<T> tripletList;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (std::uint64_t i = 0; i < dim; ++i) {
        for (std::uint64_t j = 0; j < dim; ++j) {
            if (dis(gen) > border) {
                double real = dis(gen);
                double imag = dis(gen);
                tripletList.push_back(T(i, j, StdComplex(real, imag)));
            }
        }
    }

    SparseComplexMatrix mat(dim, dim);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());

    return mat;
}

void update_by_coo(std::uint64_t target_mask,
                   std::uint64_t control_mask,
                   const SparseComplexMatrix& matrix,
                   StateVector& state) {
    SparseMatrix mat = SparseMatrix(matrix);
    auto values = mat._values;

    Kokkos::View<Complex*> update(Kokkos::ViewAllocateWithoutInitializing("update"), state.dim());
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(std::uint64_t i) {
            if ((i | control_mask) == i) {
                update(i) = 0;
            } else {
                update(i) = state._raw(i);
            }
        });
    Kokkos::fence();

    std::uint64_t outer_mask =
        ~target_mask & ((1ULL << state.n_qubits()) - 1);  // target qubit 以外の mask
    Kokkos::View<Complex*, Kokkos::MemoryTraits<Kokkos::Atomic>> update_atomic(update);
    Kokkos::parallel_for(
        "COO_Update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0, 0}, {state.dim() >> std::popcount(target_mask | control_mask), values.size()}),
        // outer: indices except operand qubit
        // inner: indices corrspond to values of given sparse matrix
        KOKKOS_LAMBDA(std::uint64_t outer, std::uint64_t inner) {
            std::uint64_t basis =
                internal::insert_zero_at_mask_positions(outer, target_mask | control_mask) |
                control_mask;
            auto [v, r, c] = values(inner);
            uint32_t src_index = internal::insert_zero_at_mask_positions(c, outer_mask) | basis;
            uint32_t dst_index = internal::insert_zero_at_mask_positions(r, outer_mask) | basis;
            // vec = matrix * vec' という式は各 r, c に対して、 vec[r] += vec'[c] * matrix[r][c]
            // の寄与を与えることに等しい。これを各 basis と合わせて更新する
            update_atomic(dst_index) += v * state._raw(src_index);
        });
    Kokkos::fence();
    state._raw = update;
}

void test_sparse(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    const std::uint64_t max_repeat = 1;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    SparseComplexMatrix mat;
    std::vector<std::uint64_t> targets(3);
    std::vector<std::uint64_t> index_list;
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    Eigen::Matrix<StdComplex, 2, 2, Eigen::RowMajor> u1, u2, u3;
    Eigen::Matrix<StdComplex, 8, 8, Eigen::RowMajor> Umerge;
    for (std::uint64_t i = 0; i < n_qubits; i++) {
        index_list.push_back(i);
    }
    for (std::uint64_t rep = 0; rep < max_repeat; rep++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_before = state.amplitudes();
        auto state_cp = state.amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        targets[2] = index_list[2];
        u1 = make_sparse_complex_matrix(1, 0.2);
        u2 = make_sparse_complex_matrix(1, 0.2);
        u3 = make_sparse_complex_matrix(1, 0.2);
        std::vector<std::uint64_t> target_list = {targets[0], targets[1], targets[2]};
        std::vector<std::uint64_t> control_list = {};
        std::sort(target_list.begin(), target_list.end());

        test_state = get_expanded_eigen_matrix_with_identity(target_list[2], u3, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(target_list[1], u2, n_qubits) *
                     get_expanded_eigen_matrix_with_identity(target_list[0], u1, n_qubits) *
                     test_state;

        Umerge = internal::kronecker_product(u3, internal::kronecker_product(u2, u1));
        mat = Umerge.sparseView();
        std::uint64_t target_mask = vector_to_mask(target_list);
        std::uint64_t control_mask = vector_to_mask(control_list);
        update_by_coo(target_mask, control_mask, mat, state);
        state_cp = state.amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            std::cout << "diff: " << std::abs((StdComplex)state_cp[i] - test_state[i])
                      << ", state_cp[" << i << "]: " << state_cp[i] << ", test_state[" << i
                      << "]: " << test_state[i] << ", bef[" << i << "]: " << state_before[i]
                      << std::endl;
        }
        for (int i = 0; i < 3; i++) {
            std::cout << "target_list[" << i << "]: " << target_list[i] << std::endl;
        }
    }
}

void run() {
    std::uint64_t n_qubits = 4;
    test_sparse(n_qubits);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
