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

    Circuit<Prec, Space> circuit(n);
    std::uint64_t target, target_sub;
    double angle;
    StdComplex imag_unit(0, 1);

    target = random.int32() % n;
    circuit.add_gate(gate::X<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_X(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Y(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::H<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::S<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_S(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Prec, Space>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_S().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::T<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_T(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Prec, Space>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_T().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtX(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Prec, Space>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtX().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtY(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Prec, Space>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtY().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::P0<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P0(), n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_gate(gate::P1<Prec, Space>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P1(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Prec, Space>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RX(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Prec, Space>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RY(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Prec, Space>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RZ(angle), n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Prec, Space>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CX(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Prec, Space>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Prec, Space>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_Swap(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U1<Prec, Space>(target, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U2<Prec, Space>(target, 0, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::U3<Prec, Space>(target, -angle, 0, 0));
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

    StateVector<Prec, Space> state = StateVector<Prec, Space>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    ComplexVector state_eigen(dim);
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec, Space> circuit(n);
    std::uint64_t target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_gate(gate::X<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::H<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::S<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::T<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Prec, Space>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Prec, Space>(target));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Prec, Space>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Prec, Space>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Prec, Space>(target, angle));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Prec, Space>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Prec, Space>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Prec, Space>(target, target_sub));

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
