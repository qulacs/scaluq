#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <gate/gate.hpp>
#include <gate/gate_factory.hpp>
#include <numbers>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

const auto eps = 1e-12;
using CComplex = std::complex<double>;

template <Gate (*QuantumGateConstructor)()>
void run_random_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const Gate gate = QuantumGateConstructor();
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(double, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = std::numbers::pi * random.uniform();
        const Gate gate = QuantumGateConstructor(angle, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = std::polar(1., angle) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(std::uint64_t, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits,
                           std::function<Eigen::MatrixXcd()> matrix_factory) {
    const auto matrix = matrix_factory();
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const std::uint64_t target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

template <Gate (*QuantumGateConstructor)(std::uint64_t, double, const std::vector<std::uint64_t>&)>
void run_random_gate_apply(std::uint64_t n_qubits,
                           std::function<Eigen::MatrixXcd(double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        const double angle = std::numbers::pi * random.uniform();
        const auto matrix = matrix_factory(angle);
        const std::uint64_t target = random.int64() % n_qubits;
        const Gate gate = QuantumGateConstructor(target, angle, {});
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        test_state = get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_IBMQ(
    std::uint64_t n_qubits,
    std::function<Eigen::MatrixXcd(double, double, double)> matrix_factory) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int gate_type = 0; gate_type < 3; gate_type++) {
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            double theta = std::numbers::pi * random.uniform();
            double phi = std::numbers::pi * random.uniform();
            double lambda = std::numbers::pi * random.uniform();
            if (gate_type == 0) {
                theta = 0;
                phi = 0;
            } else if (gate_type == 1) {
                theta = std::numbers::pi / 2;
            }
            const auto matrix = matrix_factory(theta, phi, lambda);
            const std::uint64_t target = random.int64() % n_qubits;
            Gate gate;
            if (gate_type == 0) {
                gate = gate::U1(target, lambda, {});
            } else if (gate_type == 1) {
                gate = gate::U2(target, phi, lambda, {});
            } else {
                gate = gate::U3(target, theta, phi, lambda, {});
            }
            gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();

            test_state =
                get_expanded_eigen_matrix_with_identity(target, matrix, n_qubits) * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }
}

void run_random_gate_apply_two_target(std::uint64_t n_qubits) {
    const int dim = 1ULL << n_qubits;
    Random random;

    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    std::function<Eigen::MatrixXcd(std::uint64_t, std::uint64_t, std::uint64_t)> func_eig;
    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        for (int g = 0; g < 2; g++) {
            Gate gate;
            auto state_cp = state.get_amplitudes();
            for (int i = 0; i < dim; i++) {
                test_state[i] = state_cp[i];
            }

            std::uint64_t target = random.int64() % n_qubits;
            std::uint64_t control = random.int64() % n_qubits;
            if (target == control) target = (target + 1) % n_qubits;
            if (g == 0) {
                gate = gate::CX(control, target);
                func_eig = get_eigen_matrix_full_qubit_CX;
            } else {
                gate = gate::CZ(control, target);
                func_eig = get_eigen_matrix_full_qubit_CZ;
            }
            gate->update_quantum_state(state);
            state_cp = state.get_amplitudes();

            Eigen::MatrixXcd test_mat = func_eig(control, target, n_qubits);
            test_state = test_mat * test_state;

            for (int i = 0; i < dim; i++) {
                ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
            }
        }
    }

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        for (int i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::uint64_t target1 = random.int64() % n_qubits;
        std::uint64_t target2 = random.int64() % n_qubits;
        if (target1 == target2) target1 = (target1 + 1) % n_qubits;
        auto gate = gate::Swap(target1, target2);
        gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        Eigen::MatrixXcd test_mat = get_eigen_matrix_full_qubit_Swap(target1, target2, n_qubits);
        test_state = test_mat * test_state;

        for (int i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
    }
}

void run_random_gate_apply_pauli(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    Eigen::MatrixXcd matrix;

    // Test for PauliGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        auto state_bef = state.copy();

        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }

        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
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
        for (int i = 1; i < (int)n_qubits; i++) {
            if (pauli_id_vec[i] == 0) {
                matrix = internal::kronecker_product(make_I(), matrix);
            } else if (pauli_id_vec[i] == 1) {
                matrix = internal::kronecker_product(make_X(), matrix);
            } else if (pauli_id_vec[i] == 2) {
                matrix = internal::kronecker_product(make_Y(), matrix);
            } else if (pauli_id_vec[i] == 3) {
                matrix = internal::kronecker_product(make_Z(), matrix);
            }
        }

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = gate::Pauli(pauli);
        pauli_gate->update_quantum_state(state);

        state_cp = state.get_amplitudes();
        test_state = matrix * test_state;

        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }

        auto state_bef_cp = state_bef.get_amplitudes();
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.get_amplitudes();

        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
        }
    }

    // Test for PauliRotationGate
    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.get_amplitudes();
        auto state_bef = state.copy();
        assert(test_state.size() == (int)state_cp.size());
        for (std::uint64_t i = 0; i < dim; i++) {
            test_state[i] = state_cp[i];
        }
        const double angle = std::numbers::pi * random.uniform();
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
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
        for (int i = 1; i < (int)n_qubits; i++) {
            if (pauli_id_vec[i] == 0) {
                matrix = internal::kronecker_product(make_I(), matrix);
            } else if (pauli_id_vec[i] == 1) {
                matrix = internal::kronecker_product(make_X(), matrix);
            } else if (pauli_id_vec[i] == 2) {
                matrix = internal::kronecker_product(make_Y(), matrix);
            } else if (pauli_id_vec[i] == 3) {
                matrix = internal::kronecker_product(make_Z(), matrix);
            }
        }
        matrix = std::cos(angle / 2) * Eigen::MatrixXcd::Identity(dim, dim) -
                 StdComplex(0, 1) * std::sin(angle / 2) * matrix;
        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate pauli_gate = gate::PauliRotation(pauli, angle);
        pauli_gate->update_quantum_state(state);
        state_cp = state.get_amplitudes();
        test_state = matrix * test_state;
        assert((int)state_cp.size() == test_state.size());
        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)state_cp[i] - test_state[i]), 0, eps);
        }
        Gate pauli_gate_inv = pauli_gate->get_inverse();
        pauli_gate_inv->update_quantum_state(state);
        state_cp = state.get_amplitudes();
        auto state_bef_cp = state_bef.get_amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(std::abs((CComplex)(state_cp[i] - state_bef_cp[i])), 0, eps);
        }
    }
}

TEST(GateTest, ApplyI) { run_random_gate_apply<gate::I>(5); }
TEST(GateTest, ApplyGlobalPhase) { run_random_gate_apply<gate::GlobalPhase>(5); }
TEST(GateTest, ApplyX) { run_random_gate_apply<gate::X>(5, make_X); }
TEST(GateTest, ApplyY) { run_random_gate_apply<gate::Y>(5, make_Y); }
TEST(GateTest, ApplyZ) { run_random_gate_apply<gate::Z>(5, make_Z); }
TEST(GateTest, ApplyH) { run_random_gate_apply<gate::H>(5, make_H); }
TEST(GateTest, ApplyS) { run_random_gate_apply<gate::S>(5, make_S); }
TEST(GateTest, ApplySdag) { run_random_gate_apply<gate::Sdag>(5, make_Sdag); }
TEST(GateTest, ApplyT) { run_random_gate_apply<gate::T>(5, make_T); }
TEST(GateTest, ApplyTdag) { run_random_gate_apply<gate::Tdag>(5, make_Tdag); }
TEST(GateTest, ApplySqrtX) { run_random_gate_apply<gate::SqrtX>(5, make_SqrtX); }
TEST(GateTest, ApplySqrtY) { run_random_gate_apply<gate::SqrtY>(5, make_SqrtY); }
TEST(GateTest, ApplySqrtXdag) { run_random_gate_apply<gate::SqrtXdag>(5, make_SqrtXdag); }
TEST(GateTest, ApplySqrtYdag) { run_random_gate_apply<gate::SqrtYdag>(5, make_SqrtYdag); }
TEST(GateTest, ApplyP0) { run_random_gate_apply<gate::P0>(5, make_P0); }
TEST(GateTest, ApplyP1) { run_random_gate_apply<gate::P1>(5, make_P1); }
TEST(GateTest, ApplyRX) { run_random_gate_apply<gate::RX>(5, make_RX); }
TEST(GateTest, ApplyRY) { run_random_gate_apply<gate::RY>(5, make_RY); }
TEST(GateTest, ApplyRZ) { run_random_gate_apply<gate::RZ>(5, make_RZ); }

TEST(GateTest, ApplyIBMQ) { run_random_gate_apply_IBMQ(5, make_U); }

TEST(GateTest, ApplyTwoTarget) { run_random_gate_apply_two_target(5); }

TEST(GateTest, ApplyPauliGate) { run_random_gate_apply_pauli(5); }

TEST(GateTest, ApplyProbablisticGate) {
    auto probgate = gate::Probablistic({.1, .9}, {gate::X(0), gate::I()});
    std::uint64_t x_cnt = 0, i_cnt = 0;
    StateVector state(1);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        std::uint64_t before = state.sampling(1)[0];
        probgate->update_quantum_state(state);
        std::uint64_t after = state.sampling(1)[0];
        if (before != after) {
            x_cnt++;
        } else {
            i_cnt++;
        }
    }
    // These test is probablistic, but pass at least 99.99% cases.
    ASSERT_GT(x_cnt, 0);
    ASSERT_GT(i_cnt, 0);
    ASSERT_LT(x_cnt, i_cnt);
}

void test_gate(Gate gate_control,
               Gate gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask) {
    StateVector state = StateVector::Haar_random_state(n_qubits);
    auto amplitudes = state.get_amplitudes();
    StateVector state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<Complex> amplitudes_controlled(state_controlled.dim());
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) | control_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state);
    gate_simple->update_quantum_state(state_controlled);
    amplitudes = state.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        ASSERT_NEAR(
            Kokkos::abs(amplitudes_controlled[i] -
                        amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                   control_mask]),
            0.,
            eps);
    }
}

template <std::uint64_t num_target, std::uint64_t num_rotation, typename Factory>
void test_standard_gate_control(Factory factory, std::uint64_t n) {
    Random random;
    std::vector<std::uint64_t> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::vector<std::uint64_t> targets(num_target);
    for (std::uint64_t i : std::views::iota(0ULL, num_target)) {
        targets[i] = shuffled[i];
    }
    std::uint64_t num_control = random.int32() % (n - num_target + 1);
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[num_target + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    std::vector<double> angles(num_rotation);
    for (double& angle : angles) angle = random.uniform() * std::numbers::pi * 2;
    if constexpr (num_target == 0 && num_rotation == 1) {
        Gate g1 = factory(angles[0], controls);
        Gate g2 = factory(angles[0], {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 0) {
        Gate g1 = factory(targets[0], controls);
        Gate g2 =
            factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 1) {
        Gate g1 = factory(targets[0], angles[0], controls);
        Gate g2 = factory(
            targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)), angles[0], {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 2) {
        Gate g1 = factory(targets[0], angles[0], angles[1], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          angles[0],
                          angles[1],
                          {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 1 && num_rotation == 3) {
        Gate g1 = factory(targets[0], angles[0], angles[1], angles[2], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          angles[0],
                          angles[1],
                          angles[2],
                          {});
        test_gate(g1, g2, n, control_mask);
    } else if constexpr (num_target == 2 && num_rotation == 0) {
        Gate g1 = factory(targets[0], targets[1], controls);
        Gate g2 = factory(targets[0] - std::popcount(control_mask & ((1ULL << targets[0]) - 1)),
                          targets[1] - std::popcount(control_mask & ((1ULL << targets[1]) - 1)),
                          {});
        test_gate(g1, g2, n, control_mask);
    } else {
        FAIL();
    }
}

template <bool rotation>
void test_pauli_control(std::uint64_t n) {
    PauliOperator::Data data1, data2;
    std::vector<std::uint64_t> controls;
    std::uint64_t control_mask = 0;
    std::uint64_t num_control = 0;
    Random random;
    for (std::uint64_t i : std::views::iota(0ULL, n)) {
        std::uint64_t dat = random.int32() % 12;
        if (dat < 4) {
            data1.add_single_pauli(i, dat);
            data2.add_single_pauli(i - num_control, dat);
        } else if (dat < 8) {
            controls.push_back(i);
            control_mask |= 1ULL << i;
            num_control++;
        }
    }
    if constexpr (!rotation) {
        Gate g1 = gate::Pauli(PauliOperator(data1), controls);
        Gate g2 = gate::Pauli(PauliOperator(data2), {});
        test_gate(g1, g2, n, control_mask);
    } else {
        double angle = random.uniform() * std::numbers::pi * 2;
        Gate g1 = gate::PauliRotation(PauliOperator(data1), angle, controls);
        Gate g2 = gate::PauliRotation(PauliOperator(data2), angle, {});
        test_gate(g1, g2, n, control_mask);
    }
}

TEST(GateTest, Control) {
    std::uint64_t n = 10;
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, 10)) {
        test_standard_gate_control<0, 1>(gate::GlobalPhase, n);
        test_standard_gate_control<1, 0>(gate::X, n);
        test_standard_gate_control<1, 0>(gate::Y, n);
        test_standard_gate_control<1, 0>(gate::Z, n);
        test_standard_gate_control<1, 0>(gate::S, n);
        test_standard_gate_control<1, 0>(gate::Sdag, n);
        test_standard_gate_control<1, 0>(gate::T, n);
        test_standard_gate_control<1, 0>(gate::Tdag, n);
        test_standard_gate_control<1, 0>(gate::SqrtX, n);
        test_standard_gate_control<1, 0>(gate::SqrtXdag, n);
        test_standard_gate_control<1, 0>(gate::SqrtY, n);
        test_standard_gate_control<1, 0>(gate::SqrtYdag, n);
        test_standard_gate_control<1, 0>(gate::P0, n);
        test_standard_gate_control<1, 0>(gate::P1, n);
        test_standard_gate_control<1, 1>(gate::RX, n);
        test_standard_gate_control<1, 1>(gate::RY, n);
        test_standard_gate_control<1, 1>(gate::RZ, n);
        test_standard_gate_control<1, 1>(gate::U1, n);
        test_standard_gate_control<1, 2>(gate::U2, n);
        test_standard_gate_control<1, 3>(gate::U3, n);
        test_standard_gate_control<2, 0>(gate::Swap, n);
        test_pauli_control<false>(n);
        test_pauli_control<true>(n);
    }
}
