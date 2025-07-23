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
void merge_gate_test() {
    std::vector<Gate<Prec, Space>> gates;
    Random random;
    std::uint64_t n = 4;

    auto make_control_values = [&](std::uint64_t length) {
        std::uint64_t control_value_mask = random.int64();
        std::vector<std::uint64_t> control_values(length);
        for (std::uint64_t i = 0; i < length; ++i) control_values[i] = control_value_mask >> i & 1;
        return control_values;
    };

    auto none_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::vector<std::uint64_t> controls(shuffled.begin(), shuffled.begin() + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(random.uniform() * std::numbers::pi * 2, controls, control_values));
        }
    };
    auto single_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target, controls, control_values));
        }
    };
    auto single_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(
                fac(target, random.uniform() * std::numbers::pi * 2, controls, control_values));
        }
    };
    auto single_target_rotation2 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls,
                                control_values));
        }
    };
    auto single_target_rotation3 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls,
                                control_values));
        }
    };
    auto double_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target0 = shuffled[0];
            std::uint64_t target1 = shuffled[1];
            std::vector<std::uint64_t> controls(shuffled.begin() + 2, shuffled.begin() + 2 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target0, target1, controls, control_values));
        }
    };
    auto dense_matrix = [&](auto fac) {
        for (auto nt : {0, 1, 2, 3}) {
            for (auto nc : {0, 1, 2}) {
                if (nt + nc > static_cast<int>(n)) continue;
                std::vector<std::uint64_t> shuffled = random.permutation(n);
                std::vector<uint64_t> targets(shuffled.begin(), shuffled.begin() + nt);
                std::vector<std::uint64_t> controls(shuffled.begin() + nt,
                                                    shuffled.begin() + nt + nc);
                ComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        mat(i, j) = StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                    }
                }
                std::vector<std::uint64_t> control_values = make_control_values(nc);
                gates.push_back(fac(targets, mat, controls, control_values, false));
            }
        }
    };
    auto sparse_matrix = [&](auto fac) {
        for (auto nt : {0, 1, 2, 3}) {
            for (auto nc : {0, 1, 2}) {
                if (nt + nc > static_cast<int>(n)) continue;
                std::vector<std::uint64_t> shuffled = random.permutation(n);
                std::vector<uint64_t> targets(shuffled.begin(), shuffled.begin() + nt);
                std::vector<std::uint64_t> controls(shuffled.begin() + nt,
                                                    shuffled.begin() + nt + nc);
                SparseComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        if (random.uniform() < .5) {
                            mat.insert(i, j) =
                                StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                        }
                    }
                }
                std::vector<std::uint64_t> control_values = make_control_values(nc);
                gates.push_back(fac(targets, mat, controls, control_values));
            }
        }
    };
    // gates.push_back(gate::I<Prec, Space>());
    // none_target_rotation(gate::GlobalPhase<Prec, Space>);
    single_target(gate::X<Prec, Space>);
    // single_target(gate::Y<Prec, Space>);
    // single_target(gate::Z<Prec, Space>);
    // single_target(gate::H<Prec, Space>);
    // single_target(gate::S<Prec, Space>);
    // single_target(gate::Sdag<Prec, Space>);
    // single_target(gate::T<Prec, Space>);
    // single_target(gate::Tdag<Prec, Space>);
    // single_target(gate::SqrtX<Prec, Space>);
    // single_target(gate::SqrtXdag<Prec, Space>);
    // single_target(gate::SqrtY<Prec, Space>);
    // single_target(gate::SqrtYdag<Prec, Space>);
    // single_target(gate::P0<Prec, Space>);
    // single_target(gate::P1<Prec, Space>);
    // single_target_rotation(gate::RX<Prec, Space>);
    // single_target_rotation(gate::RY<Prec, Space>);
    // single_target_rotation(gate::RZ<Prec, Space>);
    // single_target_rotation(gate::U1<Prec, Space>);
    // single_target_rotation2(gate::U2<Prec, Space>);
    // single_target_rotation3(gate::U3<Prec, Space>);
    // double_target(gate::Swap<Prec, Space>);
    // dense_matrix(gate::DenseMatrix<Prec, Space>);
    // sparse_matrix(gate::SparseMatrix<Prec, Space>);
    gates.push_back(gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("X 0", 1)));
    gates.push_back(gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("Z 0", 1)));
    gates.push_back(gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("Z 3", 1), {1}));
    // gates.push_back(gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>(n, "Z 1", 1), {0, 3}));
    // gates.push_back(
    //     gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>(n, "X 0 Y 2",
    //     random.uniform()),
    //                                      random.uniform() * std::numbers::pi * 2));
    // gates.push_back(
    //     gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>(n, "Z 0", random.uniform()),
    //                                      random.uniform() * std::numbers::pi * 2));
    // gates.push_back(
    //     gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>(n, "Z 3", random.uniform()),
    //                                      random.uniform() * std::numbers::pi * 2,
    //                                      {1}));
    // gates.push_back(
    //     gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>(n, "Z 1", random.uniform()),
    //                                      random.uniform() * std::numbers::pi * 2,
    //                                      {0, 3}));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            auto state1 = StateVector<Prec, Space>::Haar_random_state(n);
            auto state2 = state1.copy();
            auto [mg, phase] = merge_gate(g1, g2);
            g1->update_quantum_state(state1);
            g2->update_quantum_state(state1);
            mg->update_quantum_state(state2);
            state2.multiply_coef(std::polar(1., phase));
            auto state1_cp = state1.get_amplitudes();
            auto state2_cp = state2.get_amplitudes();
            for (std::uint64_t i = 0; i < state1.dim(); ++i) {
                if (std::abs(state1_cp[i] - state2_cp[i]) > 1e-6) {
                    std::cerr << g1 << std::endl;
                    std::cerr << g1->get_matrix() << std::endl;
                    std::cerr << g2 << std::endl;
                    std::cerr << g2->get_matrix() << std::endl;
                    std::cerr << mg << std::endl;
                    std::cerr << mg->get_matrix() << std::endl;
                    throw std::runtime_error(
                        "merge_gate_test: state1 and state2 are not equal after merge_gate");
                }
            }
        }
    }
}

int main() {
    scaluq::initialize();  // 初期化

    merge_gate_test<Precision::F64, ExecutionSpace::Default>();

    scaluq::finalize();  // 終了処理
}
