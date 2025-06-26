#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;
inline ComplexMatrix kronecker_product(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
    ComplexMatrix result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
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
template <Precision Prec, ExecutionSpace Space>
std::pair<Operator<Prec, Space>, Eigen::MatrixXcd> generate_random_observable_with_eigen(
    std::uint64_t n, Random& random) {
    std::uint64_t dim = 1ULL << n;
    Operator<Prec, Space> rand_observable(n);
    Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

    std::uint64_t term_count = 1;
    for (std::uint64_t term = 0; term < term_count; ++term) {
        std::vector<std::uint64_t> paulis(n, 0);
        Eigen::MatrixXcd test_rand_operator_term = Eigen::MatrixXcd::Identity(dim, dim);
        double coef = random.uniform();
        for (std::uint64_t i = 0; i < paulis.size(); ++i) {
            paulis[i] = random.int32() % 4;

            test_rand_operator_term *= get_expanded_eigen_matrix_with_identity(
                i, get_eigen_matrix_single_Pauli(paulis[i]), n);
        }
        test_rand_observable += coef * test_rand_operator_term;

        std::string str = "";
        for (std::uint64_t ind = 0; ind < paulis.size(); ind++) {
            std::uint64_t val = paulis[ind];
            if (val == 0)
                str += " I";
            else if (val == 1)
                str += " X";
            else if (val == 2)
                str += " Y";
            else if (val == 3)
                str += " Z";
            str += " " + std::to_string(ind);
        }
        rand_observable.add_operator(PauliOperator<Prec, Space>(str.c_str(), coef));
        std::cout << str << std::endl;
    }
    return {std::move(rand_observable), std::move(test_rand_observable)};
}

template <Precision Prec, ExecutionSpace Space>
void test() {
    std::uint64_t n = 3;
    std::uint64_t dim = 1ULL << n;
    Random random;
    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen<Prec, Space>(n, random);

        auto matrix = rand_observable.get_matrix();
        // for (std::uint64_t i = 0; i < dim; i++) {
        //     for (std::uint64_t j = 0; j < dim; j++) {
        //         // ASSERT_NEAR(matrix(i, j).real(), test_rand_observable(i, j).real(),
        //         eps<Prec>); if (std::abs(matrix(i, j).real() - test_rand_observable(i, j).real())
        //         > 1e-6) {
        //             std::cerr << test_rand_observable(i, j).real() << ' ' << matrix(i, j).real()
        //                       << std::endl;
        //         }
        //         if (std::abs(matrix(i, j).imag() - test_rand_observable(i, j).imag()) > 1e-6) {
        //             std::cerr << test_rand_observable(i, j).imag() << ' ' << matrix(i, j).imag()
        //                       << std::endl;
        //         }
        //     }
        // }
        std::cout << matrix << std::endl;
        std::cout << "-------------------\n";
        std::cout << test_rand_observable << std::endl;
        std::cout << "===================\n";
    }
}

int main() {
    scaluq::initialize();  // 初期化

    test<Precision::F64, ExecutionSpace::Host>();  // テスト実行

    scaluq::finalize();  // 終了処理
}
