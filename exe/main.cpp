#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../qulacs/all.hpp"
#include "../tests/gate/util.hpp"

using namespace qulacs;
using namespace std;

const auto eps = 1e-12;
using CComplex = std::complex<double>;

void run_random_gate_apply_pauli(UINT);

void run() { run_random_gate_apply_pauli(3); }

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}

void run_random_gate_apply_pauli(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::MatrixXcd matrix;

    // Test for PauliGate
    for (int repeat = 0; repeat < 1; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        auto state_bef = state.copy();

        for (UINT i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::vector<UINT> target_vec, pauli_id_vec;
        for (UINT target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        if (pauli_id_vec[0] == 0) {
            matrix = make_I();
        } else if (pauli_id_vec[0] == 1) {
            matrix = make_X();
        } else if (pauli_id_vec[0] == 2) {
            matrix = make_Y();
        } else if (pauli_id_vec[0] == 3) {
            matrix = make_Z();
        }
        for (int i = 1; i <= n_qubits; i++) {
            if (pauli_id_vec[i] == 0) {
                matrix = kronecker_product(matrix, make_I());
            } else if (pauli_id_vec[i] == 1) {
                matrix = kronecker_product(matrix, make_X());
            } else if (pauli_id_vec[i] == 2) {
                matrix = kronecker_product(matrix, make_Y());
            } else if (pauli_id_vec[i] == 3) {
                matrix = kronecker_product(matrix, make_Z());
            }
        }

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = Pauli(&pauli);
        pauli_gate->update_quantum_state(state);

        state_cp = state.amplitudes();
        // matrix = Eigen::MatrixXcd::Identity(dim, dim) - Complex(0, 1) * matrix;
        test_state = matrix * test_state;

        // check if the state is updated correctly
        for (UINT i = 0; i < dim; i++) {
            // ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
            cout << "state[" << i << "]: " << (CComplex)state_cp[i] << ", test[" << i
                 << "]: " << test_state[i] << endl;
        }

        auto state_bef_cp = state_bef.amplitudes();
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.amplitudes();

        // check if the state is restored correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
        }
    }

    // Test for PauliRotationGate
    // for (int repeat = 0; repeat < 10; repeat++) {
    //     std::cout << "repeat: " << repeat << std::endl;
    //     auto state = StateVector::Haar_random_state(n_qubits);
    //     auto state_cp = state.amplitudes();
    //     auto state_bef = state.copy();

    //     for (UINT i = 0; i < dim; i++) {
    //         test_state[i] = state_cp[i];
    //     }

    //     const double angle = M_PI * random.uniform();
    //     std::vector<UINT> target_vec, pauli_id_vec;
    //     for (UINT target = 0; target < n_qubits; target++) {
    //         target_vec.emplace_back(target);
    //         pauli_id_vec.emplace_back(random.int64() % 4);
    //     }

    //     if (pauli_id_vec[n_qubits - 1] == 0) {
    //         matrix = make_I();
    //     } else if (pauli_id_vec[n_qubits - 1] == 1) {
    //         matrix = make_X();
    //     } else if (pauli_id_vec[n_qubits - 1] == 2) {
    //         matrix = make_Y();
    //     } else if (pauli_id_vec[n_qubits - 1] == 3) {
    //         matrix = make_Z();
    //     }
    //     for (int i = n_qubits - 2; i >= 0; i--) {
    //         if (pauli_id_vec[i] == 0) {
    //             matrix = kronecker_product(make_I(), matrix);
    //         } else if (pauli_id_vec[i] == 1) {
    //             matrix = kronecker_product(make_X(), matrix);
    //         } else if (pauli_id_vec[i] == 2) {
    //             matrix = kronecker_product(make_Y(), matrix);
    //         } else if (pauli_id_vec[i] == 3) {
    //             matrix = kronecker_product(make_Z(), matrix);
    //         }
    //     }

    //     matrix = std::cos(angle / 2) * Eigen::MatrixXcd::Identity(dim, dim) -
    //              Complex(0, 1) * std::sin(angle / 2) * matrix;
    //     PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
    //     Gate pauli_gate = PauliRotation(&pauli, angle);
    //     pauli_gate->update_quantum_state(state);

    //     state_cp = state.amplitudes();
    //     test_state = matrix * test_state;
    //     // check if the state is updated correctly
    //     for (UINT i = 0; i < dim; i++) {
    //         // ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
    //         std::cout << "assertion i:" << i << ", diff: ";
    //         std::cout << std::abs((CComplex)state_cp[i] - test_state[i]) << std::endl;
    //     }

    //     Gate pauli_gate_inv = pauli_gate->get_inverse();
    //     pauli_gate_inv->update_quantum_state(state);
    //     state_cp = state.amplitudes();
    //     auto state_bef_cp = state_bef.amplitudes();
    //     // check if the state is restored correctly
    //     for (UINT i = 0; i < dim; i++) {
    //         // ASSERT_NEAR(std::abs((CComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
    //         std::cout << "assertion i:" << i << ", diff: ";
    //         std::cout << std::abs((CComplex)(state_cp[i] - state_bef_cp[i])) << std::endl;
    //     }
    // }
}
