#include <gtest/gtest.h>

#include <array>
#include <random>
#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>
#include <unsupported/Eigen/MatrixFunctions>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class CircuitTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(CircuitTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
void circuit_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector state = StateVector<Prec, Space>::Haar_random_state(n);
    ComplexVector state_eigen(dim);

    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec> circuit;
    std::uint64_t target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_gate(gate::X<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_X(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Y(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::H<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::S<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_S(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Prec>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_S().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::T<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_T(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Prec>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_T().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtX(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Prec>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtX().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtY(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Prec>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtY().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::P0<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P0(), n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_gate(gate::P1<Prec>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P1(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Prec>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RX(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Prec>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RY(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Prec>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RZ(angle), n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Prec>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CX(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Prec>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Prec>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_Swap(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U1<Prec>(target, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U2<Prec>(target, 0, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::U3<Prec>(target, -angle, 0, 0));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_U(-angle, 0, 0), n) * state_eigen;

    circuit.update_quantum_state(state);

    state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) {
        check_near<Prec>(state_eigen[i], state_cp[i]);
    }
}

TYPED_TEST(CircuitTest, CircuitBasic) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    circuit_test<Prec, Space>();
}

template <Precision Prec, ExecutionSpace Space>
void circuit_rev_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    auto state = StateVector<Prec, Space>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    ComplexVector state_eigen(dim);
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec> circuit;
    std::uint64_t target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_gate(gate::X<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::H<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::S<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::T<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Prec>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Prec>(target));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Prec>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Prec>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Prec>(target, angle));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Prec>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Prec>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Prec>(target, target_sub));

    circuit.update_quantum_state(state);

    auto revcircuit = circuit.get_inverse();

    revcircuit.update_quantum_state(state);
    state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) {
        check_near<Prec>(state_eigen[i], state_cp[i]);
    }
}

TYPED_TEST(CircuitTest, CircuitRev) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    circuit_rev_test<Prec, Space>();
}

TYPED_TEST(CircuitTest, ThrowsWhenStateHasTooFewQubits) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(1));

    StateVector<Prec, Space> state(1);
    ASSERT_THROW(circuit.update_quantum_state(state), std::runtime_error);
}

template <Precision Prec, ExecutionSpace Space>
void circuit_suzuki_trotter_test() {
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        // skip suzuki trotter for f16 and bf16
        GTEST_SKIP()
            << "F16 and BF16 are skipped due to the limitation of floating-point precision in "
               "Suzuki Trotter decomposition.";
    }
    const std::uint64_t n = 2;
    const std::uint64_t dim = 1ULL << n;
    double suzuki_trotter_eps = 1e-2;
    if constexpr (Prec == Precision::F32) {
        suzuki_trotter_eps = 1e-1;
    }
    std::vector<double> coef;
    Random random(1918);
    ComplexMatrix I = make_I();
    ComplexMatrix X = make_X();
    ComplexMatrix Y = make_Y();
    ComplexMatrix Z = make_Z();

    auto state = StateVector<Prec, Space>::Haar_random_state(n, 1918);
    auto state_cp = state.get_amplitudes();
    ComplexVector state_eigen(dim);
    for (uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec> circuit;

    for (uint64_t i = 0; i < 6; ++i) {
        coef.push_back(-random.uniform());
    }
    double angle = 2 * random.uniform() * 3.141592653589793;
    const auto num_repeats = static_cast<uint64_t>(std::ceil(angle * (double)n * 100.));
    std::vector<PauliOperator<Prec>> terms;
    terms.emplace_back(PauliOperator<Prec>("Z 0 I 1", coef[0]));
    terms.emplace_back(PauliOperator<Prec>("X 0 Y 1", coef[1]));
    terms.emplace_back(PauliOperator<Prec>("Z 0 Z 1", coef[2]));
    terms.emplace_back(PauliOperator<Prec>("Z 0 X 1", coef[3]));
    terms.emplace_back(PauliOperator<Prec>("Y 0 X 1", coef[4]));
    terms.emplace_back(PauliOperator<Prec>("I 0 Z 1", coef[5]));

    Operator<Prec, Space> observable(terms);
    circuit.add_observable_rotation_gate(observable, angle, num_repeats);

    ComplexMatrix test_observable = coef[0] * scaluq::internal::kronecker_product(I, Z);
    test_observable += coef[1] * scaluq::internal::kronecker_product(Y, X);
    test_observable += coef[2] * scaluq::internal::kronecker_product(Z, Z);
    test_observable += coef[3] * scaluq::internal::kronecker_product(X, Z);
    test_observable += coef[4] * scaluq::internal::kronecker_product(X, Y);
    test_observable += coef[5] * scaluq::internal::kronecker_product(Z, I);

    ComplexMatrix test_circuit = StdComplex(0, -1) * angle * 0.5 * test_observable;
    test_circuit = test_circuit.eval().exp();

    circuit.update_quantum_state(state);
    state_eigen = test_circuit * state_eigen;
    auto res = observable.get_expectation_value(state);
    auto test_res = (state_eigen.adjoint() * test_observable * state_eigen)(0, 0);

    ASSERT_NEAR(abs(test_res.real() - res.real()) / test_res.real(), 0, suzuki_trotter_eps);

    state.set_zero_state();
    state_cp = state.get_amplitudes();
    for (uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    circuit.update_quantum_state(state);
    state_eigen = test_circuit * state_eigen;
    res = observable.get_expectation_value(state);
    test_res = (state_eigen.adjoint() * test_observable * state_eigen)(0, 0);

    ASSERT_NEAR(abs(test_res.real() - res.real()) / test_res.real(), 0, suzuki_trotter_eps);
}

TYPED_TEST(CircuitTest, CircuitSuzukiTrotter) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    circuit_suzuki_trotter_test<Prec, Space>();
}

TYPED_TEST(CircuitTest, UpdateQuantumStateStoresMeasurementInClassicalRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 3));

    StateVector<Prec, Space> state(1);
    ClassicalRegister classical_register(4);
    circuit.update_quantum_state(state, classical_register, {}, 0);

    EXPECT_TRUE(classical_register[3]);

    const auto amplitudes = state.get_amplitudes();
    check_near<Prec>(amplitudes[0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1], StdComplex{1.0, 0.0});
}

TYPED_TEST(CircuitTest, AddConditionalGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::Measurement<Prec>(0, 0, true));  // must be reg[0]=0
    circuit.add_conditional_gate(gate::X<Prec>(0), 0, 0);   // must be |01>
    circuit.add_conditional_gate(gate::X<Prec>(1), 0, 1);   // must not be |10>

    StateVector<Prec, Space> state(2);
    ClassicalRegister classical_register(1);
    circuit.update_quantum_state(state, classical_register, {}, 0);
    EXPECT_FALSE(classical_register[0]);

    const auto amplitudes = state.get_amplitudes();
    check_near<Prec>(amplitudes[0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1], StdComplex{1.0, 0.0});
    check_near<Prec>(amplitudes[2], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[3], StdComplex{0.0, 0.0});
}

TYPED_TEST(CircuitTest, AddConditionalGateBatched) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::Measurement<Prec>(0, 0, true));
    circuit.add_conditional_gate(gate::X<Prec>(0), 0, 0);
    circuit.add_conditional_gate(gate::X<Prec>(1), 0, 1);

    StateVectorBatched<Prec, Space> states(2, 2);
    states.load({
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
    });
    ClassicalRegisterBatched classical_register(1, 2);
    circuit.update_quantum_state(states, classical_register, {}, 0);

    EXPECT_FALSE(classical_register[0][0]);
    EXPECT_TRUE(classical_register[1][0]);

    const auto amplitudes = states.get_amplitudes();
    check_near<Prec>(amplitudes[0][0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[0][1], StdComplex{1.0, 0.0});
    check_near<Prec>(amplitudes[0][2], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[0][3], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1][0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1][1], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1][2], StdComplex{1.0, 0.0});
    check_near<Prec>(amplitudes[1][3], StdComplex{0.0, 0.0});
}

TYPED_TEST(CircuitTest, ThrowsWhenMeasurementRequiresClassicalRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::Measurement<Prec>(0, 0));

    StateVector<Prec, Space> state(1);
    EXPECT_THROW(circuit.update_quantum_state(state, {}, 0), std::runtime_error);

    StateVectorBatched<Prec, Space> states(2, 1);
    EXPECT_THROW(circuit.update_quantum_state(states, {}, 0), std::runtime_error);
}

TYPED_TEST(CircuitTest, ThrowsWhenConditionalGateRequiresClassicalRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_conditional_gate(gate::X<Prec>(0), 0, true);

    StateVector<Prec, Space> state(1);
    EXPECT_THROW(circuit.update_quantum_state(state, {}, 0), std::runtime_error);

    StateVectorBatched<Prec, Space> states(2, 1);
    EXPECT_THROW(circuit.update_quantum_state(states, {}, 0), std::runtime_error);
}

TYPED_TEST(CircuitTest, UpdateQuantumStateBatchedStoresMeasurementInClassicalRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 1));

    StateVectorBatched<Prec, Space> states(2, 1);
    ClassicalRegisterBatched classical_register(2, 2);

    ASSERT_NO_THROW(circuit.update_quantum_state(states, classical_register, {}, 0));
    EXPECT_TRUE(classical_register[0][1]);
    EXPECT_TRUE(classical_register[1][1]);
}

TYPED_TEST(CircuitTest, ReusesMeasuredQubitForGatesAndRepeatedMeasurements) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::Measurement<Prec>(0, 0));
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 1));
    circuit.add_conditional_param_gate(gate::ParamRX<Prec>(0), "theta", 1, true);
    circuit.add_gate(gate::Measurement<Prec>(0, 2));

    StateVector<Prec, Space> state(1);
    ClassicalRegister classical_register(3);

    circuit.update_quantum_state(state, classical_register, {{"theta", std::numbers::pi}}, 0);
    EXPECT_FALSE(classical_register[0]);
    EXPECT_TRUE(classical_register[1]);
    EXPECT_FALSE(classical_register[2]);
    EXPECT_NEAR(std::abs(state.get_amplitudes()[0]), 1., eps<Prec>);

    StateVectorBatched<Prec, Space> states(2, 1);
    ClassicalRegisterBatched registers(3, 2);
    circuit.update_quantum_state(
        states, registers, {{"theta", {std::numbers::pi, std::numbers::pi}}}, 0);
    for (std::uint64_t i = 0; i < 2; ++i) {
        EXPECT_FALSE(registers[i][0]);
        EXPECT_TRUE(registers[i][1]);
        EXPECT_FALSE(registers[i][2]);
        EXPECT_NEAR(std::abs(states.get_amplitudes()[i][0]), 1., eps<Prec>);
    }
}

TYPED_TEST(CircuitTest, AllowsReusingQubitAfterResettingMeasurement) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::Measurement<Prec>(0, 0, true));
    circuit.add_gate(gate::X<Prec>(0));

    StateVector<Prec, Space> state(1);
    ClassicalRegister classical_register(1);

    ASSERT_NO_THROW(circuit.update_quantum_state(state, classical_register, {}, 0));
}

TYPED_TEST(CircuitTest, OptimizeDoesNotMergeMeasurementGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 0));

    circuit.template optimize<Space>();

    ASSERT_EQ(circuit.n_gates(), 2);
    EXPECT_EQ(std::get<0>(circuit.get_gate_at(0)).gate_type(), GateType::X);
    EXPECT_EQ(std::get<0>(circuit.get_gate_at(1)).gate_type(), GateType::Measurement);
}

TYPED_TEST(CircuitTest, SimulateNoiseBatchedMatchesIndependentInputs) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    constexpr std::uint64_t shots = 128;
    constexpr std::uint64_t seed = 17;
    const std::vector<double> angles{0.27, 0.71};

    StateVectorBatched<Prec, Space> inputs(2, 1);
    StateVector<Prec, Space> one(1);
    one.set_computational_basis(1);
    inputs.set_state_vector_at(1, one);
    const auto original = inputs.get_amplitudes();

    Circuit<Prec> circuit;
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_param_gate(
        gate::ParamProbabilistic<Prec>({0.4, 0.6}, {gate::I<Prec>(), gate::ParamRX<Prec>(0)}),
        "angle");
    // A sparse branch replaces its state allocation and must still update the packed output.
    circuit.add_gate(gate::Probabilistic<Prec>(
        {1.}, {gate::SparseMatrix<Prec, Space>({0}, make_X().sparseView())}));
    const auto result = circuit.simulate_noise(inputs, shots, {{"angle", angles}}, seed);
    ASSERT_EQ(result.input_indices.size(), result.states.batch_size());
    ASSERT_EQ(result.counts.size(), result.states.batch_size());
    const auto amplitudes = result.states.get_amplitudes();
    for (std::uint64_t input = 0; input < inputs.batch_size(); ++input) {
        auto scalar = circuit.simulate_noise(
            inputs.view_state_vector_at(input), shots, {{"angle", angles[input]}}, seed + input);
        std::mt19937 random_engine(seed + input);
        const std::vector<double> deterministic{1.};
        std::discrete_distribution<std::uint64_t> fixed(deterministic.begin(), deterministic.end());
        for (std::uint64_t i = 0; i < shots; ++i) (void)fixed(random_engine);
        const std::vector<double> probabilities{0.4, 0.6};
        std::discrete_distribution<std::uint64_t> alternative(probabilities.begin(),
                                                              probabilities.end());
        std::array<std::uint64_t, 2> expected_counts{};
        for (std::uint64_t i = 0; i < shots; ++i) ++expected_counts[alternative(random_engine)];

        ASSERT_EQ(scalar.size(), expected_counts.size());
        std::size_t branch = 0;
        for (std::size_t i = 0; i < result.input_indices.size(); ++i) {
            if (result.input_indices[i] != input) continue;
            ASSERT_LT(branch, scalar.size());
            EXPECT_EQ(result.counts[i], expected_counts[branch]);
            EXPECT_EQ(scalar[branch].second, expected_counts[branch]);
            const auto expected_amplitudes = scalar[branch].first.get_amplitudes();
            for (std::size_t j = 0; j < expected_amplitudes.size(); ++j)
                EXPECT_NEAR(std::abs(amplitudes[i][j] - expected_amplitudes[j]), 0., eps<Prec>);
            ++branch;
        }
        EXPECT_EQ(branch, scalar.size());
        const auto second = scalar[1].first.get_amplitudes();
        scalar[0].first.set_zero_norm_state();
        EXPECT_EQ(scalar[1].first.get_amplitudes(), second);
    }
    EXPECT_EQ(inputs.get_amplitudes(), original);
}

TYPED_TEST(CircuitTest, SimulateNoisePreservesZeroShotsAndParameterValidation) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    StateVector<Prec, Space> state(1);
    StateVectorBatched<Prec, Space> states(2, 1);
    Circuit<Prec> circuit;

    const StateVector<Prec, Space> uninitialized;
    EXPECT_THROW(circuit.simulate_noise(uninitialized, 0), std::runtime_error);
    const auto unchanged = circuit.simulate_noise(state, 0);
    ASSERT_EQ(unchanged.size(), 1);
    EXPECT_EQ(unchanged.front().second, 0);
    EXPECT_EQ(unchanged.front().first.get_amplitudes(), state.get_amplitudes());
    EXPECT_EQ(circuit.simulate_noise(states, 0).states.batch_size(), 0);

    circuit.add_gate(gate::X<Prec>(0));
    EXPECT_TRUE(circuit.simulate_noise(state, 0).empty());
    EXPECT_EQ(circuit.simulate_noise(states, 0).states.batch_size(), 0);

    circuit.add_param_gate(
        gate::ParamProbabilistic<Prec>({1., 0.}, {gate::I<Prec>(), gate::ParamRX<Prec>(0)}),
        "angle");
    EXPECT_THROW(circuit.simulate_noise(state, 1), std::out_of_range);
    EXPECT_THROW(circuit.simulate_noise(states, 1), std::out_of_range);
    EXPECT_THROW(circuit.simulate_noise(states, 1, {{"angle", {0.27}}}), std::invalid_argument);
}
