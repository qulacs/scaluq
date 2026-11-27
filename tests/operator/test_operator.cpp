#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdlib>
#include <random>
#include <scaluq/operator/operator.hpp>
#include <string>

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
    std::vector<PauliOperator<Prec>> rand_observable;
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
        rand_observable.push_back(PauliOperator<Prec>(str.c_str(), coef));
    }
    return {Operator<Prec, Space>(rand_observable), std::move(test_rand_observable)};
}

TYPED_TEST(OperatorTest, Hermitian) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::vector<PauliOperator<Prec>> terms = {
        PauliOperator<Prec>("X 0 Y 2", StdComplex(1, 0)),
        PauliOperator<Prec>("Y 1 X 3", StdComplex(2, 0)),
    };
    Operator<Prec, Space> op(terms);
    EXPECT_TRUE(op.is_hermitian());
    op *= StdComplex(0, 1);
    EXPECT_FALSE(op.is_hermitian());
    op *= StdComplex(0, -1);
    op.force_hermitian();
    EXPECT_TRUE(op.is_hermitian());
}

TYPED_TEST(OperatorTest, ArnoldiGroundStateRejectsNonHermitianOperator) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Operator<Prec, Space> op({PauliOperator<Prec>("X 0", 1.)});
    op *= StdComplex(0, 1);
    StateVector<Prec, Space> initial_state = StateVector<Prec, Space>::Haar_random_state(1, 0);

    ASSERT_THROW(
        {
            [[maybe_unused]] auto ground_state =
                op.solve_ground_state_by_arnoldi_method(initial_state, 2);
        },
        std::runtime_error);
}

TYPED_TEST(OperatorTest, GroundStateSolversRejectInvalidInput) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    using Op = Operator<Prec, Space>;
    using State = StateVector<Prec, Space>;

    const auto expect_invalid = [](const char* input,
                                   const Op& op,
                                   const State& state,
                                   std::uint64_t iterations,
                                   std::optional<StdComplex> shift = std::nullopt,
                                   std::optional<double> tolerance = std::nullopt) {
        SCOPED_TRACE(input);
        EXPECT_THROW(
            (void)op.solve_ground_state_by_power_method(state, iterations, shift, tolerance),
            std::invalid_argument);
        EXPECT_THROW(
            (void)op.solve_ground_state_by_arnoldi_method(state, iterations, shift, tolerance),
            std::invalid_argument);
    };

    const Op op({PauliOperator<Prec>("Z 0", 1.)});
    const Op out_of_range({PauliOperator<Prec>("Z 1", 1.)});
    const State state(1);
    State zero_state(1);
    zero_state.set_zero_norm_state();

    expect_invalid("zero iterations", op, state, 0);
    expect_invalid("zero tolerance", op, state, 2, std::nullopt, 0.);
    EXPECT_THROW((void)op.solve_ground_state_by_arnoldi_method(state, 2, StdComplex(0., 1.)),
                 std::invalid_argument);
    expect_invalid("qubit out of range", out_of_range, state, 2);
    expect_invalid("zero initial state", op, zero_state, 2);
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

TYPED_TEST(OperatorTest, ParallelExpectationValue) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;

    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        auto [rand_observable, test_rand_observable] =
            generate_random_observable_with_eigen<Prec, Space>(n, random);

        auto state = StateVector<Prec, Space>::Haar_random_state(n);
        auto res_parallel = rand_observable.get_expectation_values(state);
        std::vector<PauliOperator<Prec>> terms = rand_observable.get_terms();
        for (std::uint64_t term_id = 0; term_id < terms.size(); ++term_id) {
            auto pauli = terms[term_id];
            auto res = pauli.get_expectation_value(state);
            ASSERT_NEAR(res.real(), res_parallel[term_id].real(), eps<Prec>);
            ASSERT_NEAR(res.imag(), res_parallel[term_id].imag(), eps<Prec>);
        }
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

TYPED_TEST(OperatorTest, SpaceConversionCreatesIndependentCopy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    Random random(0);
    auto op = generate_random_observable_with_eigen<Prec, Space>(4, random).first;

    auto op_default = op.copy_to_default_space();
    auto op_host = op.copy_to_host_space();

    ASSERT_EQ(op.to_string(), op_default.to_string());
    ASSERT_EQ(op.to_string(), op_host.to_string());
    ASSERT_EQ(op.is_hermitian(), op_default.is_hermitian());
    ASSERT_EQ(op.is_hermitian(), op_host.is_hermitian());

    op_default *= StdComplex(2., 0.);
    ASSERT_NE(op.to_string(), op_default.to_string());
    ASSERT_EQ(op.to_string(), op_host.to_string());
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
    state_vector.load([] {
        std::vector<StdComplex> tmp(1 << n_qubits);
        for (std::uint64_t i = 0; i < tmp.size(); ++i) tmp[i] = StdComplex(i, 0);
        return tmp;
    }());

    std::vector<PauliOperator<Prec>> terms = {{0b001, 0b010, StdComplex(2)}, {"X 2 Y 1", 1}};
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
    std::vector<PauliOperator<Prec>> terms = {{"X 0 Y 1", 1.},
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

// Keep randomized coverage while making a failure reproducible with its reported seed.
inline std::uint64_t ground_state_test_seed() {
    if (const char* value = std::getenv("SCALUQ_GROUND_STATE_TEST_SEED")) {
        return std::stoull(value);
    }
    return std::random_device{}();
}

template <Precision Prec, ExecutionSpace Space>
void check_ground_state(const Operator<Prec, Space>& op,
                        const typename Operator<Prec, Space>::GroundState& ground_state,
                        StdComplex minimum_eigenvalue) {
    const double scale = op.calculate_default_mu().real();
    const double residual_tolerance = (Prec == Precision::F64 ? 1e-10 : 1e-5) * scale;
    ASSERT_TRUE(ground_state.converged);
    ASSERT_LE(std::abs(ground_state.eigenvalue - minimum_eigenvalue), residual_tolerance);
    ASSERT_NEAR(ground_state.state.get_squared_norm(), 1., eps<Prec>);
    auto applied = ground_state.state.copy();
    op.apply_to_state(applied);
    const auto applied_amplitudes = applied.get_amplitudes();
    const auto amplitudes = ground_state.state.get_amplitudes();
    double squared_residual = 0.;
    for (std::uint64_t i = 0; i < applied.dim(); ++i) {
        squared_residual +=
            std::norm(applied_amplitudes[i] - ground_state.eigenvalue * amplitudes[i]);
    }
    const double residual = std::sqrt(squared_residual);
    ASSERT_LE(residual, residual_tolerance);
    ASSERT_NEAR(ground_state.residual_norm, residual, eps<Prec> * scale);
}

template <Precision Prec, ExecutionSpace Space>
void check_random_arnoldi_ground_state() {
    const std::uint64_t seed = ground_state_test_seed();
    Random random(seed);
    for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
        SCOPED_TRACE(::testing::Message() << "seed=" << seed << ", repeat=" << repeat);
        std::uint64_t n = random.int32() % 4 + 3;
        auto [op, eigen] = generate_random_observable_with_eigen<Prec, Space>(n, random);
        Eigen::ComplexEigenSolver<ComplexMatrix> solver(eigen);
        ASSERT_EQ(solver.info(), Eigen::ComputationInfo::Success);
        auto eigenvalues = solver.eigenvalues();
        StdComplex minimum_eigenvalue = *std::ranges::min_element(
            eigenvalues, [](const auto& l, const auto& r) { return l.real() < r.real(); });
        StateVector<Prec, Space> initial_state =
            StateVector<Prec, Space>::Haar_random_state(n, random.int64());
        const auto ground_state = op.solve_ground_state_by_arnoldi_method(
            initial_state, initial_state.dim());
        ASSERT_NO_FATAL_FAILURE(check_ground_state(op, ground_state, minimum_eigenvalue));
    }
}

TYPED_TEST(OperatorTest, GroundStatePowerMethod) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        // Ground-state accuracy is not guaranteed in the performance-oriented low-precision modes.
        GTEST_SKIP() << "Ground-state accuracy is not tested for F16/BF16.";
    } else {
        // The ground-state eigenvalue is -1.75 and the spectral gap is 0.5.
        const Operator<Prec, Space> op({{"X 0", 1.}, {"Z 1", 0.5}, {"Y 2", -0.25}});
        StateVector<Prec, Space> state(3);
        state.load(std::vector<StdComplex>{1., 0., 1., 0., 0., 0., 0., 0.});
        check_ground_state(op, op.solve_ground_state_by_power_method(state, 1000), -1.75);
    }
}

TYPED_TEST(OperatorTest, GroundStateArnoldiMethod) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        // Safety, including Krylov breakdown, is checked separately for all precisions below.
        GTEST_SKIP() << "Randomized ground-state accuracy is not tested for F16/BF16.";
    } else {
        check_random_arnoldi_ground_state<Prec, Space>();
    }
}

TYPED_TEST(OperatorTest, PowerMethodUsesCustomTolerance) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const Operator<Prec, Space> op({{"Z 0", 1.}});
    StateVector<Prec, Space> state(1);
    state.load(std::vector<StdComplex>{1., 1.});

    // After one iteration the residual is sqrt(5)/3, versus 0.6 if the imaginary part is dropped.
    const StdComplex shift(2., 1.);
    EXPECT_FALSE(op.solve_ground_state_by_power_method(state, 1, shift, 0.675).converged);
    EXPECT_TRUE(op.solve_ground_state_by_power_method(state, 1, shift, 1.5).converged);
}

TYPED_TEST(OperatorTest, ArnoldiMethodUsesCustomTolerance) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const Operator<Prec, Space> op({{"Z 0", 1.}});
    StateVector<Prec, Space> state(1);
    state.load(std::vector<StdComplex>{1., 1.});

    EXPECT_FALSE(op.solve_ground_state_by_arnoldi_method(state, 1, std::nullopt, 0.25).converged);
    EXPECT_TRUE(op.solve_ground_state_by_arnoldi_method(state, 1, std::nullopt, 1.5).converged);
}

TYPED_TEST(OperatorTest, PowerMethodProducesFiniteState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    // The default shift makes the shifted identity zero.
    Operator<Prec, Space> op({PauliOperator<Prec>("", 1.)});
    StateVector<Prec, Space> initial_state(0);
    initial_state.load(std::vector<StdComplex>{StdComplex(2.)});
    EXPECT_EQ(op.get_expectation_value(initial_state), StdComplex(4.));
    const auto ground_state = op.solve_ground_state_by_power_method(initial_state, 10);
    EXPECT_EQ(ground_state.eigenvalue, StdComplex(1.));
    EXPECT_TRUE(ground_state.converged);
    EXPECT_EQ(ground_state.iterations, 0);
    EXPECT_EQ(ground_state.residual_norm, 0.);
    ASSERT_TRUE(std::isfinite(ground_state.eigenvalue.real()));
    ASSERT_TRUE(std::isfinite(ground_state.eigenvalue.imag()));
    const auto squared_norm = ground_state.state.get_squared_norm();
    ASSERT_TRUE(std::isfinite(squared_norm));
    ASSERT_GT(squared_norm, 0.0);
    for (const auto& amplitude : ground_state.state.get_amplitudes()) {
        ASSERT_TRUE(std::isfinite(amplitude.real()));
        ASSERT_TRUE(std::isfinite(amplitude.imag()));
    }
}

TYPED_TEST(OperatorTest, ArnoldiGroundStateHandlesKrylovBreakdown) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Operator<Prec, Space> op({PauliOperator<Prec>("Z 0", 1.), PauliOperator<Prec>("Z 1", 1.)});
    StateVector<Prec, Space> initial_state = StateVector<Prec, Space>::Haar_random_state(2, 0);
    auto ground_state = op.solve_ground_state_by_arnoldi_method(initial_state, 20);

    ASSERT_TRUE(std::isfinite(ground_state.eigenvalue.real()));
    ASSERT_TRUE(std::isfinite(ground_state.eigenvalue.imag()));
    const auto squared_norm = ground_state.state.get_squared_norm();
    ASSERT_TRUE(std::isfinite(squared_norm));
    ASSERT_GT(squared_norm, 0.0);
    for (const auto& amplitude : ground_state.state.get_amplitudes()) {
        ASSERT_TRUE(std::isfinite(amplitude.real()));
        ASSERT_TRUE(std::isfinite(amplitude.imag()));
    }
    if constexpr (Prec == Precision::F32 || Prec == Precision::F64) {
        ASSERT_NEAR(std::abs(ground_state.eigenvalue - StdComplex(-2.)), 0, eps<Prec>);
    }

    if constexpr (Prec == Precision::F32 || Prec == Precision::F64) {
        StateVector<Prec, Space> eigenvector1 = ground_state.state.copy();
        StateVector<Prec, Space> eigenvector2 = ground_state.state.copy();
        op.apply_to_state(eigenvector1);
        eigenvector2.multiply_coef(ground_state.eigenvalue);
        auto amp1 = eigenvector1.get_amplitudes();
        auto amp2 = eigenvector2.get_amplitudes();
        for (std::uint64_t i : std::views::iota(0ULL, eigenvector1.dim())) {
            ASSERT_NEAR(std::abs(amp1[i] - amp2[i]), 0, eps<Prec>);
        }
    }
}

TYPED_TEST(OperatorTest, ArnoldiDoesNotBreakDownOnNonzeroResidual) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    // The first orthogonalized residual has norm 0.5. In BF16, the former
    // 100 * epsilon threshold (0.78125) incorrectly stopped after one basis vector.
    // In F64, also exercise a residual below the former absolute threshold.
    constexpr double coefficient = Prec == Precision::F64 ? 1e-16 : 0.5;
    Operator<Prec, Space> op({PauliOperator<Prec>("Z 0", coefficient)});
    StateVector<Prec, Space> initial_state(1);
    initial_state.load(std::vector<StdComplex>{1., 1.});
    initial_state.normalize();
    const auto ground_state = op.solve_ground_state_by_arnoldi_method(initial_state, 2);
    ASSERT_TRUE(std::isfinite(ground_state.eigenvalue.real()));
    ASSERT_NEAR(ground_state.eigenvalue.real(), -coefficient, coefficient * 0.2);
}

TYPED_TEST(OperatorTest, GroupedPauliCachedAndStreaming) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::vector<PauliOperator<Prec>> terms = {PauliOperator<Prec>("Y 0", StdComplex(2., 1.)),
                                              PauliOperator<Prec>("X 0 Z 1", StdComplex(-1., .5)),
                                              PauliOperator<Prec>("Z 1", StdComplex(.25, -.5)),
                                              PauliOperator<Prec>("", StdComplex(.5, .25))};
    StateVector<Prec, Space> ket(2), bra(2);
    ket.load(std::vector<StdComplex>{1., StdComplex(0., 1.), 3., 4.});
    bra.load(std::vector<StdComplex>{StdComplex(1., 1.), 2., -1., 3.});
    auto expected = ket.copy();
    expected.set_zero_norm_state();
    for (const auto& term : terms) {
        auto applied = ket.copy();
        term.apply_to_state(applied);
        expected.add_state_vector_with_coef(StdComplex(.5, -.25), applied);
    }
    for (std::uint64_t budget : {0ULL, 4096ULL}) {
        GroupedPauliOperator<Prec, Space> grouped(2, terms, budget);
        EXPECT_EQ(grouped.n_groups(), 2);
        EXPECT_EQ(grouped.is_cached(), budget != 0);
        auto output = ket.copy();
        output.set_zero_norm_state();
        grouped.apply_add(ket, output, StdComplex(.5, -.25));
        auto a = expected.get_amplitudes(), b = output.get_amplitudes();
        auto initial = ket.get_amplitudes();
        for (std::size_t i = 0; i < a.size(); ++i) {
            check_near<Prec>(a[i], b[i]);
        }
        StdComplex transition = 0., expectation = 0.;
        auto bra_values = bra.get_amplitudes();
        for (const auto& term : terms) {
            auto applied = ket.copy();
            term.apply_to_state(applied);
            auto values = applied.get_amplitudes();
            for (std::size_t i = 0; i < values.size(); ++i) {
                transition += std::conj(bra_values[i]) * values[i];
                expectation += std::conj(initial[i]) * values[i];
            }
        }
        check_near<Prec>(grouped.get_transition_amplitude(bra, ket), transition);
        check_near<Prec>(grouped.get_expectation_value(ket), expectation);
    }
}

TYPED_TEST(OperatorTest, GroupedPauliBoundaryCases) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    StateVector<Prec, Space> scalar(0);
    scalar.load(std::vector<StdComplex>{StdComplex(2., 1.)});
    GroupedPauliOperator<Prec, Space> empty(0, {}, 100);
    EXPECT_EQ(empty.n_groups(), 0);
    EXPECT_EQ(empty.get_expectation_value(scalar), StdComplex(0.));
    auto empty_out = scalar.copy();
    empty.apply_add(scalar, empty_out, StdComplex(3., -1.));
    check_near<Prec>(empty_out.get_amplitudes()[0], StdComplex(2., 1.));

    std::vector<PauliOperator<Prec>> identity{PauliOperator<Prec>("", StdComplex(1., 2.))};
    GroupedPauliOperator<Prec, Space> below(
        0, identity, sizeof(scaluq::internal::Complex<Prec>) - 1);
    GroupedPauliOperator<Prec, Space> exact(0, identity, sizeof(scaluq::internal::Complex<Prec>));
    EXPECT_FALSE(below.is_cached());
    EXPECT_TRUE(exact.is_cached());
    check_near<Prec>(exact.get_expectation_value(scalar), StdComplex(5., 10.));

    std::vector<PauliOperator<Prec>> hopping{PauliOperator<Prec>("X 0 X 1"),
                                             PauliOperator<Prec>("Y 0 Y 1")};
    GroupedPauliOperator<Prec, Space> grouped(2, hopping, 0);
    StateVector<Prec, Space> input(2), out(2);
    input.load(std::vector<StdComplex>{0., 1., 0., 0.});
    out.set_zero_norm_state();
    grouped.apply_add(input, out);
    check_near<Prec>(out.get_amplitudes()[2], StdComplex(2.));

    using GroupedOp = GroupedPauliOperator<Prec, Space>;
    EXPECT_THROW((GroupedOp(1, hopping, 0)), std::invalid_argument);
    EXPECT_THROW(grouped.apply_add(scalar, out), std::invalid_argument);
    EXPECT_THROW(grouped.apply_add(input, scalar), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(grouped.get_transition_amplitude(scalar, input)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(grouped.get_transition_amplitude(input, scalar)),
                 std::invalid_argument);

    // 1024 amplitudes of 1.0 would saturate at 256.0 in BF16 if accumulated in BF16.
    StateVector<Prec, Space> ones(10);
    ones.load(std::vector<StdComplex>(1024, StdComplex(1., 0.)));
    GroupedPauliOperator<Prec, Space> large_id(10, {PauliOperator<Prec>("")}, 0);
    check_near<Prec>(large_id.get_expectation_value(ones), StdComplex(1024., 0.));
}

TYPED_TEST(OperatorTest, ApplyToStateMatchesTermSum) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    StateVector<Prec, Space> input(2);
    input.load(std::vector<StdComplex>{1., 2., 3., 4.});
    std::vector<PauliOperator<Prec>> terms = {PauliOperator<Prec>("Y 0", StdComplex(2., 1.)),
                                              PauliOperator<Prec>("Z 1", StdComplex(-1., 0.5)),
                                              PauliOperator<Prec>("", StdComplex(0.25, -0.5))};
    Operator<Prec, Space> op(terms);
    auto expected = input.copy();
    expected.set_zero_norm_state();
    for (const auto& term : terms) term.apply_add(input, expected);
    auto state = input.copy();
    op.apply_to_state(state);
    auto a = expected.get_amplitudes(), b = state.get_amplitudes();
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(std::abs(a[i] - b[i]), 0., 1e-5);
    }
}
