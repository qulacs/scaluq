#include <gtest/gtest.h>

#include <operator/operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace qulacs;

std::pair<Operator, Eigen::MatrixXcd> generate_random_observable_with_eigen(UINT n,
                                                                            Random& random) {
    UINT dim = 1ULL << n;
    Operator rand_observable(n);
    Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

    UINT term_count = random.int32() % 10 + 1;
    for (UINT term = 0; term < term_count; ++term) {
        std::vector<UINT> paulis(n, 0);
        Eigen::MatrixXcd test_rand_observable_term = Eigen::MatrixXcd::Identity(dim, dim);
        double coef = random.uniform();
        for (UINT i = 0; i < paulis.size(); ++i) {
            paulis[i] = random.int32() % 4;

            test_rand_observable_term *= get_expanded_eigen_matrix_with_identity(
                i, get_eigen_matrix_single_Pauli(paulis[i]), n);
        }
        test_rand_observable += coef * test_rand_observable_term;

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

TEST(ObservableTest, CheckExpectationValue) {
    UINT n = 4;
    UINT dim = 1ULL << n;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state = StateVector::Haar_random_state(n);
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state[i] = state[i];

        Complex res = rand_observable.get_expectation_value(state);
        Complex test_res = (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(res.imag(), 0, eps);
        ASSERT_NEAR(test_res.imag(), 0, eps);
    }
}

TEST(ObservableTest, CheckTransitionAmplitude) {
    UINT n = 4;
    UINT dim = 1ULL << n;
    Random random;

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen(n, random);

        auto state_bra = StateVector::Haar_random_state(n);
        Eigen::VectorXcd test_state_bra = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state_bra[i] = state_bra[i];
        auto state_ket = StateVector::Haar_random_state(n);
        Eigen::VectorXcd test_state_ket = Eigen::VectorXcd::Zero(dim);
        for (UINT i = 0; i < dim; ++i) test_state_ket[i] = state_ket[i];

        Complex res = rand_observable.get_transition_amplitude(state_bra, state_ket);
        Complex test_res = (test_state_bra.adjoint() * test_rand_observable * test_state_ket)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(test_res.imag(), res.imag(), eps);
    }
}
