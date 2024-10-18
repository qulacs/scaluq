#include <gtest/gtest.h>

#include <operator/operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

const double eps = 1e-12;

std::pair<Operator<double>, Eigen::MatrixXcd> generate_random_observable_with_eigen(
    std::uint64_t n, Random& random) {
    std::uint64_t dim = 1ULL << n;
    Operator<double> rand_observable(n);
    Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

    std::uint64_t term_count = random.int32() % 10 + 1;
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
            if (val != 0) {
                if (val == 1)
                    str += " X";
                else if (val == 2)
                    str += " Y";
                else if (val == 3)
                    str += " Z";
                str += " " + std::to_string(ind);
            }
        }
        rand_observable.add_operator(PauliOperator<double>(str.c_str(), coef));
    }
    return {std::move(rand_observable), std::move(test_rand_observable)};
}

TEST(OperatorTest, CheckExpectationValue) {
    std::uint64_t n = 4;
    std::uint64_t dim = 1ULL << n;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state = StateVector<double>::Haar_random_state(n);
        auto state_cp = state.get_amplitudes();
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state[i] = state_cp[i];

        Complex<double> res = rand_observable.get_expectation_value(state);
        Complex<double> test_res = (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(res.imag(), 0, eps);
        ASSERT_NEAR(test_res.imag(), 0, eps);
    }
}

TEST(OperatorTest, CheckTransitionAmplitude) {
    std::uint64_t n = 4;
    std::uint64_t dim = 1ULL << n;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state_bra = StateVector<double>::Haar_random_state(n);
        auto state_bra_cp = state_bra.get_amplitudes();
        Eigen::VectorXcd test_state_bra = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state_bra[i] = state_bra_cp[i];
        auto state_ket = StateVector<double>::Haar_random_state(n);
        auto state_ket_cp = state_ket.get_amplitudes();
        Eigen::VectorXcd test_state_ket = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state_ket[i] = state_ket_cp[i];

        Complex<double> res = rand_observable.get_transition_amplitude(state_bra, state_ket);
        Complex<double> test_res =
            (test_state_bra.adjoint() * test_rand_observable * test_state_ket)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(test_res.imag(), res.imag(), eps);
    }
}

TEST(OperatorTest, AddTest) {
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto op2 = generate_random_observable_with_eigen(n, random).first;
        auto op = op1 + op2;
        auto state = StateVector<double>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 + exp2 - exp), 0, eps);
    }
}

TEST(OperatorTest, SubTest) {
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto op2 = generate_random_observable_with_eigen(n, random).first;
        auto op = op1 - op2;
        auto state = StateVector<double>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 - exp2 - exp), 0, eps);
    }
}

TEST(OperatorTest, MultiCoefTest) {
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto coef = Complex<double>(random.normal(), random.normal());
        auto op = op1 * coef;
        auto state = StateVector<double>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 * coef - exp), 0, eps);
    }
}

TEST(OperatorTest, ApplyToStateTest) {
    const std::uint64_t n_qubits = 3;
    StateVector<double> state_vector(n_qubits);
    state_vector.load([n_qubits] {
        std::vector<Complex<double>> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = Complex<double>(i, 0);
        return tmp;
    }());

    Operator<double> op(n_qubits);
    op.add_operator({0b001, 0b010, Complex<double>(2)});
    op.add_operator({"X 2 Y 1", 1});
    op.apply_to_state(state_vector);

    std::vector<Complex<double>> expected = {
        Complex<double>(2, -6),
        Complex<double>(0, -7),
        Complex<double>(-6, 4),
        Complex<double>(-4, 5),
        Complex<double>(10, -2),
        Complex<double>(8, -3),
        Complex<double>(-14, 0),
        Complex<double>(-12, 1),
    };
    ASSERT_EQ(state_vector.get_amplitudes(), expected);
}

TEST(OperatorTest, Optimize) {
    Operator<double> op(2);
    op.add_operator(PauliOperator<double>("X 0 Y 1", 1.));
    op.add_operator(PauliOperator<double>("Y 0 Z 1", 2.));
    op.add_operator(PauliOperator<double>("Z 1", 3.));
    op.add_operator(PauliOperator<double>("X 0 Y 1", 4.));
    op.add_operator(PauliOperator<double>("Z 1", 4.));
    op.add_operator(PauliOperator<double>("X 0 Y 1", 5.));
    op.optimize();
    std::vector<std::pair<std::string, Complex<double>>> expected = {
        {"X 0 Y 1", 10.}, {"Y 0 Z 1", 2.}, {"Z 1", 7.}};
    std::vector<std::pair<std::string, Complex<double>>> test;
    for (const auto& pauli : op.terms()) {
        test.emplace_back(pauli.get_pauli_string(), pauli.coef());
    }
    std::ranges::sort(expected, [](const auto& l, const auto& r) { return l.first < r.first; });
    std::ranges::sort(test, [](const auto& l, const auto& r) { return l.first < r.first; });
    ASSERT_EQ(expected.size(), test.size());
    for (std::uint64_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i].first, test[i].first);
        ASSERT_NEAR(Kokkos::abs(expected[i].second - test[i].second), 0, eps);
    }
}
