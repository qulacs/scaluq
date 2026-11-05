#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
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

template <Precision Prec, ExecutionSpace Space>
void expect_ground_state(const Operator<Prec, Space>& op,
                         const typename Operator<Prec, Space>::GroundState& result,
                         double expected,
                         double tolerance) {
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.eigenvalue.real(), expected, tolerance);
    EXPECT_NEAR(result.eigenvalue.imag(), 0., tolerance);
    EXPECT_NEAR(result.state.get_squared_norm(), 1., tolerance);
    auto residual = result.state.copy();
    op.apply_to_state(residual);
    residual.add_state_vector_with_coef(-result.eigenvalue, result.state);
    EXPECT_NEAR(std::sqrt(residual.get_squared_norm()), result.residual_norm, tolerance);
    EXPECT_LE(result.residual_norm, tolerance);
}

TYPED_TEST(OperatorTest, GroundStateKnownSpectrum) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        GTEST_SKIP() << "Accuracy checks require F32 or F64.";
    } else {
        Operator<Prec, Space> op({{"Z 0", 1.}, {"Z 1", 0.25}});
        StateVector<Prec, Space> state(2);
        state.load(std::vector<StdComplex>{1., 1., 1., 1.});  // Deliberately unnormalized.
        const double tolerance = Prec == Precision::F64 ? 1e-8 : 1e-4;
        auto power = op.solve_ground_state_by_power_method(state, 1000);
        auto arnoldi = op.solve_ground_state_by_arnoldi_method(state, 20);
        expect_ground_state(op, power, -1.25, tolerance);
        expect_ground_state(op, arnoldi, -1.25, tolerance);
        const auto shifted = op.solve_ground_state_by_arnoldi_method(state, 20, 1e8);
        expect_ground_state(op, shifted, -1.25, tolerance);
        EXPECT_LE(arnoldi.iterations, state.dim());
        EXPECT_FALSE(op.solve_ground_state_by_power_method(state, 1).converged);
        EXPECT_FALSE(op.solve_ground_state_by_arnoldi_method(state, 1).converged);
    }
}

TYPED_TEST(OperatorTest, GroundStateInputAndBreakdown) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    Operator<Prec, Space> identity({{"", 1.}});
    Operator<Prec, Space> z({{"Z 0", 1.}});
    Operator<Prec, Space> out_of_range({{"Z 1", 1.}});
    StateVector<Prec, Space> state(1);
    state.load(std::vector<StdComplex>{1., 1.});
    const auto arnoldi = z.solve_ground_state_by_arnoldi_method(state, 2);
    EXPECT_TRUE(std::isfinite(arnoldi.eigenvalue.real()));
    EXPECT_NEAR(arnoldi.eigenvalue.real(), -1., 0.1);
    const auto power = identity.solve_ground_state_by_power_method(state, 10);
    EXPECT_TRUE(power.converged);
    EXPECT_TRUE(std::isfinite(power.eigenvalue.real()));
    EXPECT_NEAR(power.eigenvalue.real(),
                1.,
                Prec == Precision::F16 || Prec == Precision::BF16 ? 0.05 : 1e-5);
    EXPECT_THROW((void)z.solve_ground_state_by_power_method(state, 0), std::invalid_argument);
    EXPECT_THROW((void)z.solve_ground_state_by_arnoldi_method(state, 0), std::invalid_argument);
    EXPECT_THROW((void)z.solve_ground_state_by_power_method(state, 1, StdComplex(0., 1.)),
                 std::invalid_argument);
    EXPECT_THROW((void)z.solve_ground_state_by_arnoldi_method(state, 1, StdComplex(0., 1.)),
                 std::invalid_argument);
    EXPECT_THROW((void)out_of_range.solve_ground_state_by_power_method(state, 1),
                 std::invalid_argument);
    EXPECT_THROW((void)out_of_range.solve_ground_state_by_arnoldi_method(state, 1),
                 std::invalid_argument);
    EXPECT_THROW((void)z.solve_ground_state_by_power_method(state, 1, std::nullopt, 0.),
                 std::invalid_argument);
    state.set_zero_norm_state();
    EXPECT_THROW((void)z.solve_ground_state_by_power_method(state, 1), std::invalid_argument);
    EXPECT_THROW((void)z.solve_ground_state_by_arnoldi_method(state, 1), std::invalid_argument);
    if constexpr (Prec == Precision::F32 || Prec == Precision::F64) {
        state.load(std::vector<StdComplex>{std::numeric_limits<double>::quiet_NaN(), 1.});
        EXPECT_THROW((void)z.solve_ground_state_by_power_method(state, 1), std::invalid_argument);
        EXPECT_THROW((void)z.solve_ground_state_by_arnoldi_method(state, 1), std::invalid_argument);
    }
}

TYPED_TEST(OperatorTest, ArnoldiScaleAndDegeneracy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    if constexpr (Prec != Precision::F64) {
        GTEST_SKIP() << "Small-scale test requires F64.";
    } else {
        StateVector<Prec, Space> state(2);
        state.load(std::vector<StdComplex>{1., 1., 1., 1.});
        Operator<Prec, Space> tiny({{"Z 0", 1e-16}});
        const auto result = tiny.solve_ground_state_by_arnoldi_method(state, 20);
        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.eigenvalue.real(), -1e-16, 1e-25);
        EXPECT_LT(result.residual_norm, 1e-25);
        EXPECT_LE(result.iterations, 2);
    }
}

TYPED_TEST(OperatorTest, ArnoldiRandomHermitianMatrices) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        GTEST_SKIP() << "Accuracy checks require F32 or F64.";
    } else {
        Random random(314159);
        for (std::uint64_t repeat = 0; repeat < 10; ++repeat) {
            SCOPED_TRACE(repeat);
            auto [op, unused] = generate_random_observable_with_eigen<Prec, Space>(3, random);
            Eigen::SelfAdjointEigenSolver<ComplexMatrix> exact(op.get_full_matrix(3));
            ASSERT_EQ(exact.info(), Eigen::Success);
            auto state = StateVector<Prec, Space>::Haar_random_state(3, random.int64());
            auto result = op.solve_ground_state_by_arnoldi_method(state, state.dim());
            const double tolerance = Prec == Precision::F64 ? 1e-8 : 1e-3;
            expect_ground_state(op, result, exact.eigenvalues()[0], tolerance);
        }
    }
}

TYPED_TEST(OperatorTest, GroundStateConvergenceDoesNotProveGlobalMinimum) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    Operator<Prec, Space> op({{"Z 0", 1.}});
    StateVector<Prec, Space> state(1);  // |0> is an excited eigenstate.
    for (const auto& result : {op.solve_ground_state_by_power_method(state, 4),
                               op.solve_ground_state_by_arnoldi_method(state, 4)}) {
        EXPECT_TRUE(result.converged);
        EXPECT_NEAR(result.eigenvalue.real(), 1., 0.01);
        EXPECT_LT(result.residual_norm, 0.01);
    }
}
