#include <gtest/gtest.h>

#include <operator/operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

const double eps = 1e-12;

std::pair<Operator, Eigen::MatrixXcd> generate_random_observable_with_eigen(UINT n,
                                                                            Random& random) {
    UINT dim = 1ULL << n;
    Operator rand_observable(n);
    Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

    UINT term_count = random.int32() % 10 + 1;
    for (UINT term = 0; term < term_count; ++term) {
        std::vector<UINT> paulis(n, 0);
        Eigen::MatrixXcd test_rand_operator_term = Eigen::MatrixXcd::Identity(dim, dim);
        double coef = random.uniform();
        for (UINT i = 0; i < paulis.size(); ++i) {
            paulis[i] = random.int32() % 4;

            test_rand_operator_term *= get_expanded_eigen_matrix_with_identity(
                i, get_eigen_matrix_single_Pauli(paulis[i]), n);
        }
        test_rand_observable += coef * test_rand_operator_term;

        std::string str = "";
        for (UINT ind = 0; ind < paulis.size(); ind++) {
            UINT val = paulis[ind];
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
        rand_observable.add_operator(PauliOperator(str.c_str(), coef));
    }
    return {std::move(rand_observable), std::move(test_rand_observable)};
}

TEST(OperatorTest, CheckExpectationValue) {
    UINT n = 4;
    UINT dim = 1ULL << n;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state = StateVector::Haar_random_state(n);
        auto state_cp = state.amplitudes();
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state[i] = state_cp[i];

        Complex res = rand_observable.get_expectation_value(state);
        Complex test_res = (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(res.imag(), 0, eps);
        ASSERT_NEAR(test_res.imag(), 0, eps);
    }
}

TEST(OperatorTest, CheckTransitionAmplitude) {
    UINT n = 4;
    UINT dim = 1ULL << n;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state_bra = StateVector::Haar_random_state(n);
        auto state_bra_cp = state_bra.amplitudes();
        Eigen::VectorXcd test_state_bra = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state_bra[i] = state_bra_cp[i];
        auto state_ket = StateVector::Haar_random_state(n);
        auto state_ket_cp = state_ket.amplitudes();
        Eigen::VectorXcd test_state_ket = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state_ket[i] = state_ket_cp[i];

        Complex res = rand_observable.get_transition_amplitude(state_bra, state_ket);
        Complex test_res = (test_state_bra.adjoint() * test_rand_observable * test_state_ket)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(test_res.imag(), res.imag(), eps);
    }
}

TEST(OperatorTest, AddTest) {
    UINT n = 4;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto op2 = generate_random_observable_with_eigen(n, random).first;
        auto op = op1 + op2;
        auto state = StateVector::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 + exp2 - exp), 0, eps);
    }
}

TEST(OperatorTest, SubTest) {
    UINT n = 4;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto op2 = generate_random_observable_with_eigen(n, random).first;
        auto op = op1 - op2;
        auto state = StateVector::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 - exp2 - exp), 0, eps);
    }
}

TEST(OperatorTest, MultiCoefTest) {
    UINT n = 4;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen(n, random).first;
        auto coef = Complex(random.normal(), random.normal());
        auto op = op1 * coef;
        auto state = StateVector::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(Kokkos::abs(exp1 * coef - exp), 0, eps);
    }
}

TEST(OperatorTest, Optimize) {
    Operator op(2);
    op.add_operator(PauliOperator("X 0 Y 1", 1.));
    op.add_operator(PauliOperator("Y 0 Z 1", 2.));
    op.add_operator(PauliOperator("Z 1", 3.));
    op.add_operator(PauliOperator("X 0 Y 1", 4.));
    op.add_operator(PauliOperator("Z 1", 4.));
    op.add_operator(PauliOperator("X 0 Y 1", 5.));
    op.optimize();
    std::vector<std::pair<std::string, Complex>> expected = {
        {"X 0 Y 1", 10.}, {"Y 0 Z 1", 2.}, {"Z 1", 7.}};
    std::vector<std::pair<std::string, Complex>> test;
    for (const auto& pauli : op.terms()) {
        test.emplace_back(pauli.get_pauli_string(), pauli.get_coef());
    }
    std::ranges::sort(expected, [](const auto& l, const auto& r) { return l.first < r.first; });
    std::ranges::sort(test, [](const auto& l, const auto& r) { return l.first < r.first; });
    ASSERT_EQ(expected.size(), test.size());
    for (UINT i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i].first, test[i].first);
        ASSERT_NEAR(Kokkos::abs(expected[i].second - test[i].second), 0, eps);
    }
}
