#include <gtest/gtest.h>

#include <Eigen/Core>
#include <circuit/circuit.hpp>
#include <gate/gate_factory.hpp>
#include <util/random.hpp>

#include "../gate/util.hpp"

using namespace qulacs;

const auto eps = 1e-12;

TEST(CircuitTest, CircuitBasic) {
    const auto Identity = make_I();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();
    const auto H = make_H();
    const auto S = make_S();
    const auto T = make_T();
    const auto sqrtX = make_sqrtX();
    const auto sqrtY = make_sqrtY();
    const auto P0 = make_P0();
    const auto P1 = make_P1();

    const UINT n = 4;
    const UINT dim = 1ULL << n;

    Random random;

    StateVector state = StateVector::Haar_random_state(n);
    Eigen::VectorXcd state_eigen(dim);
    for (int i = 0; i < dim; ++i) state_eigen[i] = state[i];

    Circuit circuit(n);
    UINT target, target_sub;
    double angle;
    std::complex<double> imag_unit(0, 1);

    target = random.int32() % n;
    circuit.add_gate(gate::X(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, X, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Y(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, Y, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Z(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, Z, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::H(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::S(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, S, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, S.adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::T(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, T, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, T.adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::sqrtX(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::sqrtXdag(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtX.adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::sqrtY(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::sqrtYdag(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, sqrtY.adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::P0(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, P0, n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_gate(gate::P1(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, P1, n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RX(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RY(target), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RZ(target), n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CNOT(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CNOT(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    /*
    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);
    state_eigen = get_eigen_matrix_full_qubit_SWAP(target, target_sub, n) * state_eigen;
    */

    /*
    target = random.int32() % n;
    circuit.add_U1_gate(target, M_PI);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, Z, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_U2_gate(target, 0, M_PI);
    state_eigen = get_expanded_eigen_matrix_with_identity(target, H, n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_U3_gate(target, -angle, 0, 0);
    state_eigen = get_expanded_eigen_matrix_with_identity(
                      target, cos(angle / 2) * Identity + imag_unit * sin(angle / 2) * Y, n) *
                  state_eigen;
  */

    /*
    std::vector<UINT> target_index_list{0, 1, 2, 3};
    std::vector<UINT> pauli_id_list{0, 1, 2, 3};
    circuit.add_multi_Pauli_gate(target_index_list, pauli_id_list);

    // add same gate == cancel above pauli gate
    PauliOperator pauli = PauliOperator("I 0 X 1 Y 2 Z 3");
    circuit.add_multi_Pauli_gate(pauli);

    ComplexMatrix mat_x(2, 2);
    target = random.int32() % n;
    mat_x << 0, 1, 1, 0;
    circuit.add_dense_matrix_gate(target, mat_x);

    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(1), n) *
        state_eigen;
    */

    circuit.update_quantum_state(state);
    for (int i = 0; i < dim; ++i)
        ASSERT_NEAR(std::abs(state_eigen[i] - (std::complex<double>)state[i]), 0, eps);
}

/*
TEST(CircuitTest, CircuitRev) {
    const UINT n = 4;
    const UINT dim = 1ULL << n;

    Random random;

    QuantumState state(n);
    Eigen::VectorXcd state_eigen(dim);

    state.set_Haar_random_state();
    for (ITYPE i = 0; i < dim; ++i) state_eigen[i] = state.data_cpp()[i];

    QuantumCircuit circuit(n);
    UINT target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_X_gate(target);

    target = random.int32() % n;
    circuit.add_Y_gate(target);

    target = random.int32() % n;
    circuit.add_Z_gate(target);

    target = random.int32() % n;
    circuit.add_H_gate(target);

    target = random.int32() % n;
    circuit.add_S_gate(target);

    target = random.int32() % n;
    circuit.add_Sdag_gate(target);

    target = random.int32() % n;
    circuit.add_T_gate(target);

    target = random.int32() % n;
    circuit.add_Tdag_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtX_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtXdag_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtY_gate(target);

    target = random.int32() % n;
    circuit.add_sqrtYdag_gate(target);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RX_gate(target, angle);
    circuit.add_RotInvX_gate(target, angle);
    circuit.add_RotX_gate(target, angle);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RY_gate(target, angle);
    circuit.add_RotInvY_gate(target, angle);
    circuit.add_RotInvY_gate(target, angle);

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_RZ_gate(target, angle);
    circuit.add_RotInvZ_gate(target, angle);
    circuit.add_RotZ_gate(target, -angle);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CNOT_gate(target, target_sub);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_CZ_gate(target, target_sub);

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_SWAP_gate(target, target_sub);

    Observable observable(n);
    angle = 2 * PI * random.uniform();

    observable.add_operator(1.0, "Z 0");
    observable.add_operator(2.0, "Z 0 Z 1");

    circuit.add_diagonal_observable_rotation_gate(observable, angle);

    circuit.update_quantum_state(&state);

    auto revcircuit = circuit.get_inverse();

    revcircuit->update_quantum_state(&state);

    for (ITYPE i = 0; i < dim; ++i) ASSERT_NEAR(abs(state_eigen[i] - state.data_cpp()[i]), 0, eps);

    delete revcircuit;
}
*/
