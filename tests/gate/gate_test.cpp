#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_one_qubit.hpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "util.hpp"

const auto eps = 1e-12;

template <class QuantumGateConstructor>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVectorCpu::Haar_random_state(n_qubits);
        for (int i = 0; i < dim; i++) {
            test_state[i] = state[i];
        }

        const UINT target = random.int64() % n_qubits;
        const QuantumGateConstructor gate(target);
        gate.update_quantum_state(state);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs(state[i] - test_state[i]), 0, eps);
        }
    }
}

TEST(GateTest, ApplySingleQubitGate) { run_random_gate_apply<PauliX>(5, make_X); }
