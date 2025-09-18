#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <scaluq/operator/operator.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class OperatorTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(OperatorTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
std::pair<Operator<Prec, Space>, Eigen::MatrixXcd> generate_random_observable_with_eigen(
    std::uint64_t n, Random& random) {
    std::uint64_t dim = 1ULL << n;
    std::vector<PauliOperator<Prec, Space>> rand_observable;
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
        rand_observable.push_back(PauliOperator<Prec, Space>(str.c_str(), coef));
    }
    return {Operator<Prec, Space>(rand_observable), std::move(test_rand_observable)};
}

TYPED_TEST(OperatorTest, GetMatrix) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    std::uint64_t dim = 1ULL << n;
    Random random;
    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen<Prec, Space>(n, random);

        auto matrix = rand_observable.get_full_matrix(n);
        for (std::uint64_t i = 0; i < dim; i++) {
            for (std::uint64_t j = 0; j < dim; j++) {
                ASSERT_NEAR(matrix(i, j).real(), test_rand_observable(i, j).real(), eps<Prec>);
                ASSERT_NEAR(matrix(i, j).imag(), test_rand_observable(i, j).imag(), eps<Prec>);
            }
        }
    }
}

TYPED_TEST(OperatorTest, CheckExpectationValue) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    std::uint64_t dim = 1ULL << n;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen<Prec, Space>(n, random);

        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto state_cp = state.get_amplitudes();
        Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state[i] = state_cp[i];

        StdComplex res = rand_observable.get_expectation_value(state);
        StdComplex test_res = (test_state.adjoint() * test_rand_observable * test_state)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), 0, eps<Prec>);
        ASSERT_NEAR(test_res.imag(), 0, eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, CheckBatchedExpectationValue) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    StateVectorBatched<Prec, Space> states(10, n);
    states.set_Haar_random_state(false);
    auto [rand_observable, _] = generate_random_observable_with_eigen<Prec, Space>(n, random);
    std::vector<StdComplex> results = rand_observable.get_expectation_value(states);

    for (int b = 0; b < 10; ++b) {
        auto state = states.get_state_vector_at(b);
        StdComplex res = rand_observable.get_expectation_value(state);
        ASSERT_NEAR(res.real(), results[b].real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), results[b].imag(), eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, CheckTransitionAmplitude) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    std::uint64_t dim = 1ULL << n;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen<Prec, Space>(n, random);

        auto state_bra = StateVector<Prec, Space>::Haar_random_state(n);
        auto state_bra_cp = state_bra.get_amplitudes();
        Eigen::VectorXcd test_state_bra = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state_bra[i] = state_bra_cp[i];
        auto state_ket = StateVector<Prec, Space>::Haar_random_state(n);
        auto state_ket_cp = state_ket.get_amplitudes();
        Eigen::VectorXcd test_state_ket = Eigen::VectorXcd::Zero(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state_ket[i] = state_ket_cp[i];

        StdComplex res = rand_observable.get_transition_amplitude(state_bra, state_ket);
        StdComplex test_res =
            (test_state_bra.adjoint() * test_rand_observable * test_state_ket)(0, 0);
        ASSERT_NEAR(test_res.real(), res.real(), eps<Prec>);
        ASSERT_NEAR(test_res.imag(), res.imag(), eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, CheckBatchedTransitionAmplitude) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    StateVectorBatched<Prec, Space> states_bra(10, n);
    StateVectorBatched<Prec, Space> states_ket(10, n);
    states_bra.set_Haar_random_state(false);
    states_ket.set_Haar_random_state(false);
    auto [rand_observable, _] = generate_random_observable_with_eigen<Prec, Space>(n, random);
    std::vector<StdComplex> results =
        rand_observable.get_transition_amplitude(states_bra, states_ket);

    for (int b = 0; b < 10; ++b) {
        auto state_bra = states_bra.get_state_vector_at(b);
        auto state_ket = states_ket.get_state_vector_at(b);
        StdComplex res = rand_observable.get_transition_amplitude(state_bra, state_ket);
        ASSERT_NEAR(res.real(), results[b].real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), results[b].imag(), eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, AddTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op2 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op = op1 + op2;
        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(std::abs(exp1 + exp2 - exp), 0, eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, SubTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op2 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto op = op1 - op2;
        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp2 = op2.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(std::abs(exp1 - exp2 - exp), 0, eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, MultiCoefTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto op1 = generate_random_observable_with_eigen<Prec, Space>(n, random).first;
        auto coef = StdComplex(random.normal(), random.normal());
        auto op = op1 * coef;
        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto exp1 = op1.get_expectation_value(state);
        auto exp = op.get_expectation_value(state);
        ASSERT_NEAR(std::abs(exp1 * coef - exp), 0, eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, MultiOperatorTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [op1, eigen1] = generate_random_observable_with_eigen<Prec, Space>(n, random);
        auto [op2, eigen2] = generate_random_observable_with_eigen<Prec, Space>(n, random);
        auto op = op1 * op2;
        auto mat = op.get_full_matrix(n);
        auto expected_eigen = eigen1 * eigen2;
        for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(mat.rows()); ++i) {
            for (std::uint64_t j = 0; j < static_cast<std::uint64_t>(mat.cols()); ++j) {
                ASSERT_NEAR(mat(i, j).real(), expected_eigen(i, j).real(), eps<Prec>);
                ASSERT_NEAR(mat(i, j).imag(), expected_eigen(i, j).imag(), eps<Prec>);
            }
        }
    }
}

TYPED_TEST(OperatorTest, ApplyToStateTest) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t n_qubits = 3;
    StateVector<Prec, Space> state_vector(n_qubits);
    state_vector.load([n_qubits] {
        std::vector<StdComplex> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = StdComplex(i, 0);
        return tmp;
    }());

    std::vector<PauliOperator<Prec, Space>> terms = {{0b001, 0b010, StdComplex(2)}, {"X 2 Y 1", 1}};
    Operator<Prec, Space> op(terms);
    op.apply_to_state(state_vector);

    std::vector<StdComplex> expected = {
        StdComplex(2, -6),
        StdComplex(0, -7),
        StdComplex(-6, 4),
        StdComplex(-4, 5),
        StdComplex(10, -2),
        StdComplex(8, -3),
        StdComplex(-14, 0),
        StdComplex(-12, 1),
    };
    ASSERT_EQ(state_vector.get_amplitudes(), expected);
}

TYPED_TEST(OperatorTest, Optimize) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::vector<PauliOperator<Prec, Space>> terms = {{"X 0 Y 1", 1.},
                                                     {"Y 0 Z 1", 2.},
                                                     {"Z 1", 3.},
                                                     {"X 0 Y 1", 4.},
                                                     {"Z 1", 4.},
                                                     {"X 0 Y 1", 5.}};
    Operator<Prec, Space> op(terms);
    op.optimize();
    std::vector<std::pair<std::string, StdComplex>> expected = {
        {"X 0 Y 1", 10.}, {"Y 0 Z 1", 2.}, {"Z 1", 7.}};
    std::vector<std::pair<std::string, StdComplex>> test;
    for (const auto& pauli : op.get_terms()) {
        test.emplace_back(pauli.get_pauli_string(), pauli.coef());
    }
    std::ranges::sort(expected, [](const auto& l, const auto& r) { return l.first < r.first; });
    std::ranges::sort(test, [](const auto& l, const auto& r) { return l.first < r.first; });
    ASSERT_EQ(expected.size(), test.size());
    for (std::uint64_t i = 0; i < expected.size(); i++) {
        ASSERT_EQ(expected[i].first, test[i].first);
        ASSERT_NEAR(std::abs(expected[i].second - test[i].second), 0, eps<Prec>);
    }
}

TYPED_TEST(OperatorTest, GroundState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        std::uint64_t n = random.int32() % 4 + 3;
        auto [op, eigen] = generate_random_observable_with_eigen<Prec, Space>(n, random);
        Eigen::ComplexEigenSolver<ComplexMatrix> solver(eigen);
        ASSERT_EQ(solver.info(), Eigen::ComputationInfo::Success);
        auto eigenvalues = solver.eigenvalues();
        StdComplex minimum_eigenvalue = *std::ranges::min_element(
            eigenvalues, [](const auto& l, const auto& r) { return l.real() < r.real(); });
        StateVector<Prec, Space> initial_state = StateVector<Prec, Space>::Haar_random_state(n);
        std::uint64_t iter_count_arnoldi;
        if constexpr (Prec == Precision::F64) {
            iter_count_arnoldi = 60;
        } else {
            iter_count_arnoldi = 20;
        }
        for (auto type : {0, 1}) {
            if (Prec == Precision::F16 && type == 1) {
                // skip arnoldi method for f16
                continue;
            }
            auto ground_state =
                type == 0
                    ? op.solve_ground_state_by_power_method(initial_state, 1000)
                    : op.solve_ground_state_by_arnoldi_method(initial_state, iter_count_arnoldi);
            ASSERT_NEAR(
                std::pow(std::abs(ground_state.eigenvalue - minimum_eigenvalue), 5), 0, eps<Prec>);
            StateVector<Prec, Space> eigenvector1 = ground_state.state.copy();
            StateVector<Prec, Space> eigenvector2 = ground_state.state.copy();
            op.apply_to_state(eigenvector1);
            eigenvector2.multiply_coef(ground_state.eigenvalue);
            auto amp1 = eigenvector1.get_amplitudes();
            auto amp2 = eigenvector2.get_amplitudes();
            for (std::uint64_t i : std::views::iota(0ULL, eigenvector1.dim())) {
                ASSERT_NEAR(std::pow(std::abs(amp1[i] - amp2[i]), 5), 0, eps<Prec>);
            }
        }
    }
}
