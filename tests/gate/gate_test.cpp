#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_one_control_one_target.hpp>
#include <gate/gate_one_qubit.hpp>
#include <gate/gate_quantum_matrix.hpp>
#include <gate/gate_two_qubit.cpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "util.hpp"

namespace qulacs {
const auto eps = 1e-12;

template <class QuantumGateConstructor>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
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

template <class QuantumGateConstructor>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd(double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int i = 0; i < dim; i++) {
            test_state[i] = state[i];
        }

        const double angle = M_PI * random.uniform();
        const auto matrix = matrix_factory(angle);
        const UINT target = random.int64() % n_qubits;
        const QuantumGateConstructor gate(target, angle);
        gate.update_quantum_state(state);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs(state[i] - test_state[i]), 0, eps);
        }
    }
}

template <class QuantumGateConstructor>
void run_random_gate_apply(UINT n_qubits,
                           std::function<Eigen::MatrixXcd(double, double, double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int i = 0; i < dim; i++) {
            test_state[i] = state[i];
        }

        double theta = M_PI * random.uniform();
        double phi = M_PI * random.uniform();
        double lambda = M_PI * random.uniform();
        if constexpr (std::is_same_v<QuantumGateConstructor, U1>) {
            theta = 0;
            phi = 0;
        } else if constexpr (std::is_same_v<QuantumGateConstructor, U2>) {
            theta = M_PI / 2;
        } else if constexpr (std::is_same_v<QuantumGateConstructor, U3>) {
        } else {
            throw std::runtime_error("Invalid gate type");
        }

        const auto matrix = matrix_factory(theta, phi, lambda);
        const UINT target = random.int64() % n_qubits;
        const U3 gate(target, theta, phi, lambda);
        gate.update_quantum_state(state);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs(state[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_two_qubit(UINT n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    QuantumGate* gate;
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int g = 0; g < 2; g++) {
            for (int i = 0; i < dim; i++) {
                test_state[i] = state[i];
            }

            UINT target = random.int64() % n_qubits;
            UINT control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = new CNOT(control, target);
                func_eig = get_eigen_matrix_full_qubit_CNOT;
            } else {
                gate = new CZ(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ;
            }
            gate->update_quantum_state(state);

            Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs(state[i] - test_state[i]), 0, eps);
            }
        }
        delete gate;
    }

    func_eig = get_eigen_matrix_full_qubit_SWAP;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int i = 0; i < dim; i++) {
            test_state[i] = state[i];
        }

        UINT target = random.int64() % n_qubits;
        UINT control = random.int64() % n_qubits;
        if (target == control) target = (target + 1) % n_qubits;
        gate = new SWAP(control, target);
        gate->update_quantum_state(state);

        Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
        test_state = test_mat * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs(state[i] - test_state[i]), 0, eps);
        }
    }
    delete gate;
}

TEST(GateTest, ApplyI) { run_random_gate_apply<I>(5, make_I); }
TEST(GateTest, ApplyX) { run_random_gate_apply<X>(5, make_X); }
TEST(GateTest, ApplyY) { run_random_gate_apply<Y>(5, make_Y); }
TEST(GateTest, ApplyZ) { run_random_gate_apply<Z>(5, make_Z); }
TEST(GateTest, ApplyH) { run_random_gate_apply<H>(5, make_H); }
TEST(GateTest, ApplyS) { run_random_gate_apply<S>(5, make_S); }
TEST(GateTest, ApplySdag) { run_random_gate_apply<Sdag>(5, make_Sdag); }
TEST(GateTest, ApplyT) { run_random_gate_apply<T>(5, make_T); }
TEST(GateTest, ApplyTdag) { run_random_gate_apply<Tdag>(5, make_Tdag); }
TEST(GateTest, ApplySqrtX) { run_random_gate_apply<sqrtX>(5, make_sqrtX); }
TEST(GateTest, ApplySqrtY) { run_random_gate_apply<sqrtY>(5, make_sqrtY); }
TEST(GateTest, ApplySqrtXdag) { run_random_gate_apply<sqrtXdag>(5, make_sqrtXdag); }
TEST(GateTest, ApplySqrtYdag) { run_random_gate_apply<sqrtYdag>(5, make_sqrtYdag); }
TEST(GateTest, ApplyP0) { run_random_gate_apply<P0>(5, make_P0); }
TEST(GateTest, ApplyP1) { run_random_gate_apply<P1>(5, make_P1); }
TEST(GateTest, ApplyRX) { run_random_gate_apply<RX>(5, make_RX); }
TEST(GateTest, ApplyRY) { run_random_gate_apply<RY>(5, make_RY); }
TEST(GateTest, ApplyRZ) { run_random_gate_apply<RZ>(5, make_RZ); }
TEST(GateTest, ApplyU1) { run_random_gate_apply<U1>(5, make_U); }
TEST(GateTest, ApplyU2) { run_random_gate_apply<U2>(5, make_U); }
TEST(GateTest, ApplyU3) { run_random_gate_apply<U3>(5, make_U); }
TEST(GateTest, ApplyTwoQubit) { run_random_gate_apply_two_qubit(5); }
}  // namespace qulacs
