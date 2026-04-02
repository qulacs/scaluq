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

    circuit.update_quantum_state(state, {}, 0);

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

    circuit.update_quantum_state(state, {}, 0);

    auto revcircuit = circuit.get_inverse();

    revcircuit.update_quantum_state(state, {}, 0);
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
    ASSERT_THROW(circuit.update_quantum_state(state, {}, 0), std::runtime_error);
}

TYPED_TEST(CircuitTest, UpdateQuantumStateStoresMeasurementInCircuitRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit(4);
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 3));

    StateVector<Prec, Space> state(1);
    circuit.update_quantum_state(state, {}, 0);

    EXPECT_TRUE(circuit.classical_register()[3]);

    const auto amplitudes = state.get_amplitudes();
    check_near<Prec>(amplitudes[0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1], StdComplex{1.0, 0.0});
}

TYPED_TEST(CircuitTest, UpdateQuantumStateExecutesAdaptiveGateFromCircuitRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit(2);
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::X<Prec>(1));
    circuit.add_gate(gate::Measurement<Prec>(0, 0));
    circuit.add_gate(gate::Measurement<Prec>(1, 1));
    circuit.add_conditional_gate(gate::X<Prec>(2), [](const ClassicalRegister& reg) {
        for (std::uint64_t bit = 0; bit < 2; ++bit) {
            if (bit >= reg.size() || !reg[bit]) return false;
        }
        return true;
    });

    StateVector<Prec, Space> state(3);
    circuit.update_quantum_state(state, {}, 0);

    EXPECT_TRUE(circuit.classical_register()[0]);
    EXPECT_TRUE(circuit.classical_register()[1]);

    const auto amplitudes = state.get_amplitudes();
    check_near<Prec>(amplitudes[0], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[1], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[2], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[3], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[4], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[5], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[6], StdComplex{0.0, 0.0});
    check_near<Prec>(amplitudes[7], StdComplex{1.0, 0.0});
}

TYPED_TEST(CircuitTest, UpdateQuantumStateBatchedStoresMeasurementInFlatRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit(2);
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 1));

    StateVectorBatched<Prec, Space> states(2, 1);
    ASSERT_NO_THROW(circuit.update_quantum_state(states, {}, 0));
    EXPECT_TRUE(circuit.classical_register()[1]);
    EXPECT_TRUE(circuit.classical_register()[3]);
}

TYPED_TEST(CircuitTest, AddCircuitKeepsLongerClassicalRegister) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> lhs(1);
    Circuit<Prec> rhs(4);
    rhs.add_gate(gate::X<Prec>(0));
    rhs.add_gate(gate::Measurement<Prec>(0, 3));

    lhs.add_circuit(rhs);

    StateVector<Prec, Space> state(1);
    lhs.update_quantum_state(state, {}, 0);

    EXPECT_TRUE(lhs.classical_register()[3]);
}

TYPED_TEST(CircuitTest, JsonRoundTripKeepsClassicalRegisterSize) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;

    Circuit<Prec> circuit(4);
    circuit.add_gate(gate::X<Prec>(0));
    circuit.add_gate(gate::Measurement<Prec>(0, 3));

    auto loaded = Json(circuit).template get<Circuit<Prec>>();

    StateVector<Prec, Space> state(1);
    ASSERT_NO_THROW(loaded.update_quantum_state(state, {}, 0));
    EXPECT_TRUE(loaded.classical_register()[3]);
}
