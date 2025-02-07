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

template <Precision Prec>
void circuit_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector state = StateVector<Prec>::Haar_random_state(n);
    ComplexVector state_eigen(dim);

    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec> circuit(n);
    std::uint64_t target, target_sub;
    double angle;
    StdComplex imag_unit(0, 1);

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
    circuit_test<Prec>();
}

template <Precision Prec>
void circuit_rev_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector<Prec> state = StateVector<Prec>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    ComplexVector state_eigen(dim);
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Prec> circuit(n);
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
    circuit_rev_test<Prec>();
}
