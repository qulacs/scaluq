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

void run_random_gate_apply_IBMQ(
    UINT n_qubits, std::function<Eigen::MatrixXcd(double, double, double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int gate_type = 0; gate_type < 3; gate_type++) {
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            double theta = M_PI * random.uniform();
            double phi = M_PI * random.uniform();
            double lambda = M_PI * random.uniform();
            const auto matrix = matrix_factory(theta, phi, lambda);
            const UINT target = random.int64() % n_qubits;
            Gate gate;
            if (gate_type == 0) {
                gate = U1(target, lambda);
            } else if (gate_type == 1) {
                gate = U2(target, phi, lambda);
            } else {
                gate = U3(target, theta, phi, lambda);
            }
            gate->update_quantum_state(state);
            state_cp = state.amplitudes();

            test_state =
                get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_gate_apply_two_qubit(UINT n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    std::function<Eigen::MatrixXcd(UINT, UINT, UINT)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int g = 0; g < 2; g++) {
            Gate gate;
            auto state_cp = state.amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state[i];
            }

            UINT target = random.int64() % n_qubits;
            UINT control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = CNOT(control, target);
                func_eig = get_eigen_matrix_full_qubit_CNOT;
            } else {
                gate = CZ(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ;
            }
            gate->update_quantum_state(state);
            state_cp = state.amplitudes();

            Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }

    func_eig = get_eigen_matrix_full_qubit_SWAP;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        UINT target = random.int64() % n_qubits;
        UINT control = random.int64() % n_qubits;
        if (target == control) target = (target + 1) % n_qubits;
        auto gate = SWAP(control, target);
        gate->update_quantum_state(state);
        state_cp = state.amplitudes();

        Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
        test_state = test_mat * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_fused(UINT n_qubits, UINT target0, UINT target1, UINT block_size) {
    const UINT dim = 1ULL << n_qubits;
    StateVector state_ref = StateVector::Haar_random_state(n_qubits);
    StateVector state = state_ref.copy();

    // update "state_ref" using SWAP gate
    for (UINT i = 0; i < block_size; i++) {
        auto swap_gate = SWAP(target0 + i, target1 + i);
        swap_gate->update_quantum_state(state_ref);
    }
    auto state_ref_cp = state_ref.amplitudes();

    auto fused_swap_gate = FusedSWAP(target0, target1, block_size);
    fused_swap_gate->update_quantum_state(state);
    auto state_cp = state.amplitudes();

    for (UINT i = 0; i < dim; i++) {
        ASSERT_NEAR(std::abs((CComplex)state_cp[i] - (CComplex)state_ref_cp[i]), 0, eps);
    }
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

TEST(GateTest, ApplyIBMQ) { run_random_gate_apply_IBMQ(5, make_U); }

TEST(GateTest, ApplyTwoQubit) { run_random_gate_apply_two_qubit(5); }
TEST(GateTest, ApplyFused) {
    UINT n_qubits = 10;
    for (UINT t0 = 0; t0 < n_qubits; t0++) {
        for (UINT t1 = 0; t1 < n_qubits; t1++) {
            if (t0 == t1) continue;
            UINT max_bs =
                std::min((t0 < t1) ? (t1 - t0) : (t0 - t1), std::min(n_qubits - t0, n_qubits - t1));
            for (UINT bs = 1; bs <= max_bs; bs++) {
                run_random_gate_apply_fused(n_qubits, t0, t1, bs);
            }
        }
    }
}
