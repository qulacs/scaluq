#include <gtest/gtest.h>

#include <operator/general_quantum_operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace qulacs;

TEST(ObservableTest, CheckExpectationValue) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;

    double coef;
    Complex res;
    Complex test_res;
    Random random;

    const auto X = make_X();

    StateVector state(n);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    test_state(0) = 1.;

    coef = random.uniform();
    GeneralQuantumOperator observable(n);
    observable.add_operator(PauliOperator("X 0", coef));
    Eigen::MatrixXcd test_observable = Eigen::MatrixXcd::Zero(dim, dim);
    test_observable += coef * get_expanded_eigen_matrix_with_identity(0, X, n);

    res = observable.get_expectation_value(state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

    state = StateVector::Haar_random_state(n);
    for (UINT i = 0; i < dim; ++i) test_state[i] = state[i];
    res = observable.get_expectation_value(state);
    test_res = (test_state.adjoint() * test_observable * test_state)(0, 0);
    ASSERT_NEAR(test_res.real(), res.real(), eps);
    ASSERT_NEAR(res.imag(), 0, eps);
    ASSERT_NEAR(test_res.imag(), 0, eps);

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        GeneralQuantumOperator rand_observable(n);
        Eigen::MatrixXcd test_rand_observable = Eigen::MatrixXcd::Zero(dim, dim);

        UINT term_count = random.int32() % 10 + 1;
        for (UINT term = 0; term < term_count; ++term) {
            std::vector<UINT> paulis(n, 0);
            Eigen::MatrixXcd test_rand_observable_term = Eigen::MatrixXcd::Identity(dim, dim);
            coef = random.uniform();
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

        state = StateVector::Haar_random_state(n);
        for (UINT i = 0; i < dim; ++i) test_state[i] = state[i];

        res = rand_observable.get_expectation_value(state);
        test_res = (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps);
        ASSERT_NEAR(res.imag(), 0, eps);
        ASSERT_NEAR(test_res.imag(), 0, eps);
    }
}
