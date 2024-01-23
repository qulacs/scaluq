#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_factory.hpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "util.hpp"

using namespace qulacs;

const auto eps = 1e-12;
using CComplex = std::complex<double>;

template <Gate (*QuantumGateConstructor)(UINT)>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const UINT target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(UINT, double)>
void run_random_gate_apply(UINT n_qubits, std::function<Eigen::MatrixXcd(double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = M_PI * random.uniform();
        const auto matrix = matrix_factory(angle);
        const UINT target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, angle);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

/*
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

        const double theta = M_PI * random.uniform();
        const double phi = M_PI * random.uniform();
        const double lambda = M_PI * random.uniform();
        const QuantumGateConstructor gate;
        if (typeid(QuantumGateConstructor) == typeid(U1)) {
            theta = 0;
            phi = 0;
            gate = QuantumGateConstructor(lambda);
        } else if (typeid(QuantumGateConstructor) == typeid(U2)) {
            theta = M_PI / 2;
            gate = QuantumGateConstructor(phi, lambda);
        } else if (typeid(QuantumGateConstructor) == typeid(U3)) {
            gate = QuantumGateConstructor(theta, phi, lambda);
        } else {
            throw std::runtime_error("Invalid gate type");
        }

        const auto matrix = matrix_factory(theta, phi, lambda);
        const UINT target = random.int64() % n_qubits;
        gate.update_quantum_state(state);

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state[i] - test_state[i]), 0, eps);
        }
    }
}
*/

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
/*
TEST(GateTest, ApplyU1) { run_random_gate_apply<U1>(5, make_U); }
TEST(GateTest, ApplyU2) { run_random_gate_apply<U2>(5, make_U); }
TEST(GateTest, ApplyU3) { run_random_gate_apply<U3>(5, make_U); }
*/
