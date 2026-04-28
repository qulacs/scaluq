#include <gtest/gtest.h>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

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
