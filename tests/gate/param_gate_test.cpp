#include <gtest/gtest.h>

#include <gate/gate_factory.hpp>
#include <gate/param_gate_factory.hpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"

using namespace scaluq;

const auto eps = 1e-12;

template <typename FactoryFixed, typename FactoryParametric>
void test_apply_parametric_single_pauli_rotation(std::uint64_t n_qubits,
                                                 FactoryFixed factory_fixed,
                                                 FactoryParametric factory_parametric) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();

        const std::uint64_t target = random.int32() % n_qubits;
        const double param = M_PI * random.uniform();
        const double param_coef = random.uniform() * 2 - 1;
        const Gate gate = factory_fixed(target, param_coef * param, {});
        const ParamGate pgate = factory_parametric(target, param_coef, {});
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();

        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }

        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_bef_amp[i] - state_amp[i]), 0, eps);
        }
    }
}

void test_apply_parametric_multi_pauli_rotation(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();
        const double param = M_PI * random.uniform();
        const double param_coef = random.uniform() * 2 - 1;
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate gate = gate::PauliRotation(pauli, param_coef * param);
        ParamGate pgate = gate::ParamPauliRotation(pauli, param_coef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();
        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }
        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs((state_bef_amp[i] - state_amp[i])), 0, eps);
        }
    }
}

TEST(ParamGateTest, ApplyParamRXGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::ParamRX);
}
TEST(ParamGateTest, ApplyParamRYGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::ParamRX);
}
TEST(ParamGateTest, ApplyParamRZGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::ParamRX);
}
TEST(ParamGateTest, ApplyParamPauliRotationGate) { test_apply_parametric_multi_pauli_rotation(5); }

TEST(ParamGateTest, ApplyParamProbablisticGate) {
    auto probgate = gate::ParamProbablistic({.1, .9}, {gate::ParamRX(0), gate::I()});
    std::uint64_t x_cnt = 0, i_cnt = 0;
    StateVector state(1);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        std::uint64_t before = state.sampling(1)[0];
        probgate->update_quantum_state(state, scaluq::PI());
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

void test_gate(ParamGate gate_control,
               ParamGate gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask,
               double param) {
    StateVector state = StateVector::Haar_random_state(n_qubits);
    auto amplitudes = state.get_amplitudes();
    StateVector state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<Complex> amplitudes_controlled(state_controlled.dim());
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) | control_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state, param);
    gate_simple->update_quantum_state(state_controlled, param);
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

template <typename Factory>
void test_param_rotation_control(Factory factory, std::uint64_t n) {
    std::cerr << "prx" << std::endl;
    Random random;
    std::vector<std::uint64_t> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (std::uint64_t i : std::views::iota(0ULL, n) | std::views::reverse) {
        std::uint64_t j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    std::uint64_t target = shuffled[0];
    std::uint64_t num_control = random.int32() % n;
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[1 + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    double param = random.uniform() * PI() * 2;
    ParamGate g1 = factory(target, 1., controls);
    ParamGate g2 = factory(target - std::popcount(control_mask & ((1ULL << target) - 1)), 1., {});
    test_gate(g1, g2, n, control_mask, param);
}

void test_ppauli_control(std::uint64_t n) {
    std::cerr << "ppauli" << std::endl;
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
    double param = random.uniform() * PI() * 2;
    ParamGate g1 = gate::ParamPauliRotation(PauliOperator(data1), 1., controls);
    ParamGate g2 = gate::ParamPauliRotation(PauliOperator(data2), 1., {});
    test_gate(g1, g2, n, control_mask, param);
}

TEST(ParamGateTest, Control) {
    std::uint64_t n = 10;
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, 10)) {
        test_param_rotation_control(gate::ParamRX, n);
        test_param_rotation_control(gate::ParamRY, n);
        test_param_rotation_control(gate::ParamRZ, n);
        test_ppauli_control(n);
    }
}
