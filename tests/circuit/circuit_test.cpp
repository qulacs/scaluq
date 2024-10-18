#include "circuit/circuit.hpp"

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <numbers>

#include "../util/util.hpp"
#include "gate/gate_factory.hpp"
#include "util/random.hpp"

using namespace scaluq;

const auto eps = 1e-12;
using CComplex = std::complex<double>;

TEST(CircuitTest, CircuitBasic) {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector state = StateVector<double>::Haar_random_state(n);
    Eigen::VectorXcd state_eigen(dim);

    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<double> circuit(n);
    std::uint64_t target, target_sub;
    double angle;
    std::complex<double> imag_unit(0, 1);

    target = random.int32() % n;
    circuit.add_gate(gate::X<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_X(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Y<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Y(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Z<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::H<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::S<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_S(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<double>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_S().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::T<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_T(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<double>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_T().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtX(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<double>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtX().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_SqrtY(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<double>(target));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_SqrtY().adjoint(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::P0<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P0(), n) * state_eigen;

    target = (target + 1) % n;
    circuit.add_gate(gate::P1<double>(target));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_P1(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<double>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RX(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<double>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RY(angle), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<double>(target, angle));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_RZ(angle), n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<double>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CX(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<double>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_CZ(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<double>(target, target_sub));
    state_eigen = get_eigen_matrix_full_qubit_Swap(target, target_sub, n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U1<double>(target, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_Z(), n) * state_eigen;

    target = random.int32() % n;
    circuit.add_gate(gate::U2<double>(target, 0, std::numbers::pi));
    state_eigen = get_expanded_eigen_matrix_with_identity(target, make_H(), n) * state_eigen;

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::U3<double>(target, -angle, 0, 0));
    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, make_U(-angle, 0, 0), n) * state_eigen;

    /*
    std::vector<std::uint64_t> target_index_list{0, 1, 2, 3};
    std::vector<std::uint64_t> pauli_id_list{0, 1, 2, 3};
    circuit.add_gate(multi_Pauli(target_index_list, pauli_id_list));

    // add same gate == cancel above pauli gate
    PauliOperator pauli = PauliOperator("I 0 X 1 Y 2 Z 3");
    circuit.add_gate(multi_Pauli(pauli));

    internal::ComplexMatrix<double> mat_x(2, 2);
    target = random.int32() % n;
    mat_x << 0, 1, 1, 0;
    circuit.add_gate(dense_matrix(target, mat_x));

    state_eigen =
        get_expanded_eigen_matrix_with_identity(target, get_eigen_matrix_single_Pauli(1), n) *
        state_eigen;
    */

    circuit.update_quantum_state(state);

    state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < dim; ++i)
        ASSERT_NEAR(std::abs(state_eigen[i] - (CComplex)state_cp[i]), 0, eps);
}

TEST(CircuitTest, CircuitRev) {
    const std::uint64_t n = 4;
    const std::uint64_t dim = 1ULL << n;

    Random random;

    StateVector state = StateVector<double>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    Eigen::VectorXcd state_eigen(dim);
    for (std::uint64_t i = 0; i < dim; ++i) state_eigen[i] = state_cp[i];

    Circuit<double> circuit(n);
    std::uint64_t target, target_sub;
    double angle;

    target = random.int32() % n;
    circuit.add_gate(gate::X<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Y<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Z<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::H<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::S<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Sdag<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::T<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::Tdag<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtX<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtXdag<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtY<double>(target));

    target = random.int32() % n;
    circuit.add_gate(gate::SqrtYdag<double>(target));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RX<double>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RY<double>(target, angle));

    target = random.int32() % n;
    angle = random.uniform() * 3.14159;
    circuit.add_gate(gate::RZ<double>(target, angle));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CX<double>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::CZ<double>(target, target_sub));

    target = random.int32() % n;
    target_sub = random.int32() % (n - 1);
    if (target_sub >= target) target_sub++;
    circuit.add_gate(gate::Swap<double>(target, target_sub));

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
    for (std::uint64_t i = 0; i < dim; ++i)
        ASSERT_NEAR(abs(state_eigen[i] - (CComplex)state_cp[i]), 0, eps);
}
