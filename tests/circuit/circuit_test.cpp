#include <gtest/gtest.h>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../util/util.hpp"

using namespace scaluq;

template <std::floating_point Fp>
void circuit_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector state = StateVector<Fp>::Haar_random_state(n);
    ComplexVector<Fp> state_eigen(dim);

    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Fp> circuit(n);
    std::uint64_t target, target_sub;
    Fp angle;
    std::complex<Fp> imag_unit(0, 1);

    target = random.int32() % n;
    circuit.add_gate(gate::X<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_X<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_Y<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_Z<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::H<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_H<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::S<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_S<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Fp>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity<Fp>(target, make_S<Fp>().adjoint(), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::T<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_T<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Fp>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity<Fp>(target, make_T<Fp>().adjoint(), n) *
                  state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_SqrtX<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_SqrtX<Fp>().adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_SqrtY<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_SqrtY<Fp>().adjoint(), n) *
        state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::P0<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_P0<Fp>(), n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_gate(gate::P1<Fp>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_P1<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Fp>(target, angle));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_RX<Fp>(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Fp>(target, angle));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_RY<Fp>(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Fp>(target, angle));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_RZ<Fp>(angle), n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Fp>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CX<Fp>(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Fp>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CZ<Fp>(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Fp>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_Swap<Fp>(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U1<Fp>(target, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U2<Fp>(target, 0, std::numbers::pi));
    state_eigen =
        get_expanded_eigen_matrix_with_identity<Fp>(target, make_H<Fp>(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::U3<Fp>(target, -angle, 0, 0));
    state_eigen = get_expanded_eigen_matrix_with_identity<Fp>(target, make_U<Fp>(-angle, 0, 0), n) *
                  state_eigen;

    /*
    std::vector<std::uint64_t> target_index_list{0, 1, 2, 3};
    std::vector<std::uint64_t> pauli_id_list{0, 1, 2, 3};
    circuit.add_gate(multi_Pauli(target_index_list, pauli_id_list));

    // add same gate == cancel above pauli gate
    PauliOperator pauli = PauliOperator("I 0 X 1 Y 2 Z 3");
    circuit.add_gate(multi_Pauli(pauli));

    internal::ComplexMatrix<Fp> mat_x(2, 2);
    target = random.int32() % n;
    mat_x << 0, 1, 1, 0;
    circuit.add_gate(dense_matrix(target, mat_x));

    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(1), n) *
        state_eigen;
    */

    circuit.update_quantum_state(state);

    state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) {
        check_near(state_eigen[i], (StdComplex<Fp>)state_cp[i]);
    }
}

TEST(CircuitTest, CircuitBasic) {
    circuit_test<double>();
    circuit_test<float>();
}

template <std::floating_point Fp>
void circuit_rev_test() {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector<Fp> state = StateVector<Fp>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    ComplexVector<Fp> state_eigen(dim);
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<Fp> circuit(n);
    std::uint64_t target, target_sub;
    Fp angle;

    target = random.int32() % n;
    circuit.add_gate(gate::X<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Y<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Z<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::H<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::S<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::T<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<Fp>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<Fp>(target));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<Fp>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<Fp>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<Fp>(target, angle));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<Fp>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<Fp>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<Fp>(target, target_sub));

    /*
    Observable observable(n);
    angle = 2 * std::numbers::pi * random.uniform();

    observable.add_operator(1.0, "Z 0");
    observable.add_operator(2.0, "Z 0 Z 1");

    circuit.add_gate(diagonal_observable_rotation(observable, angle));
    */

    circuit.update_quantum_state(state);

    auto revcircuit = circuit.get_inverse();

    revcircuit.update_quantum_state(state);
    state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) {
        check_near(state_eigen[i], (StdComplex<Fp>)state_cp[i]);
    }
}

TEST(CircuitTest, CircuitRev) {
    circuit_rev_test<double>();
    circuit_rev_test<float>();
}
