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
void test_apply_parametric_single_pauli_rotation(UINT n_qubits,
                                                 FactoryFixed factory_fixed,
                                                 FactoryParametric factory_parametric) {
    const UINT dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();

        const UINT target = random.int32() % n_qubits;
        const double param = M_PI * random.uniform();
        const double pcoef = random.uniform() * 2 - 1;
        const Gate gate = factory_fixed(target, pcoef * param, {});
        const ParamGate pgate = factory_parametric(target, pcoef, {});
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.amplitudes();
        auto state_cp_amp = state_cp.amplitudes();

        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }

        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.amplitudes();
        auto state_bef_amp = state_bef.amplitudes();
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_bef_amp[i] - state_amp[i]), 0, eps);
        }
    }
}

void test_apply_parametric_multi_pauli_rotation(UINT n_qubits) {
    const UINT dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();
        const double param = M_PI * random.uniform();
        const double pcoef = random.uniform() * 2 - 1;
        std::vector<UINT> target_vec, pauli_id_vec;
        for (UINT target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        PauliOperator pauli(target_vec, pauli_id_vec, 1.0);
        Gate gate = gate::PauliRotation(pauli, pcoef * param);
        ParamGate pgate = gate::PPauliRotation(pauli, pcoef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.amplitudes();
        auto state_cp_amp = state_cp.amplitudes();
        // check if the state is updated correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }
        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.amplitudes();
        auto state_bef_amp = state_bef.amplitudes();
        // check if the state is restored correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs((state_bef_amp[i] - state_amp[i])), 0, eps);
        }
    }
}

TEST(ParamGateTest, ApplyPRXGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::PRX);
}
TEST(ParamGateTest, ApplyPRYGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::PRX);
}
TEST(ParamGateTest, ApplyPRZGate) {
    test_apply_parametric_single_pauli_rotation(5, &gate::RX, &gate::PRX);
}
TEST(ParamGateTest, ApplyPPauliRotationGate) { test_apply_parametric_multi_pauli_rotation(5); }

TEST(ParamGateTest, ApplyPProbablisticGate) {
    auto probgate = gate::PProbablistic({.1, .9}, {gate::PRX(0), gate::I()});
    UINT x_cnt = 0, i_cnt = 0;
    StateVector state(1);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        UINT before = state.sampling(1)[0];
        probgate->update_quantum_state(state, scaluq::PI());
        UINT after = state.sampling(1)[0];
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

void test_gate(
    ParamGate gate_control, ParamGate gate_simple, UINT n_qubits, UINT control_mask, double param) {
    StateVector state = StateVector::Haar_random_state(n_qubits);
    auto amplitudes = state.amplitudes();
    StateVector state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<Complex> amplitudes_controlled(state_controlled.dim());
    for (UINT i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) | control_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state, param);
    gate_simple->update_quantum_state(state_controlled, param);
    amplitudes = state.amplitudes();
    amplitudes_controlled = state_controlled.amplitudes();
    for (UINT i : std::views::iota(0ULL, state_controlled.dim())) {
        ASSERT_NEAR(
            Kokkos::abs(amplitudes_controlled[i] -
                        amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                   control_mask]),
            0.,
            eps);
    }
}

template <typename Factory>
void test_param_rotation_control(Factory factory, UINT n) {
    std::cerr << "prx" << std::endl;
    Random random;
    std::vector<UINT> shuffled(n);
    std::iota(shuffled.begin(), shuffled.end(), 0ULL);
    for (UINT i : std::views::iota(0ULL, n) | std::views::reverse) {
        UINT j = random.int32() % (i + 1);
        if (i != j) std::swap(shuffled[i], shuffled[j]);
    }
    UINT target = shuffled[0];
    UINT num_control = random.int32() % n;
    std::vector<UINT> controls(num_control);
    for (UINT i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[1 + i];
    }
    UINT control_mask = 0ULL;
    for (UINT c : controls) control_mask |= 1ULL << c;
    double param = random.uniform() * PI() * 2;
    ParamGate g1 = factory(target, 1., controls);
    ParamGate g2 = factory(target - std::popcount(control_mask & ((1ULL << target) - 1)), 1., {});
    test_gate(g1, g2, n, control_mask, param);
}

void test_ppauli_control(UINT n) {
    std::cerr << "ppauli" << std::endl;
    PauliOperator::Data data1, data2;
    std::vector<UINT> controls;
    UINT control_mask = 0;
    UINT num_control = 0;
    Random random;
    for (UINT i : std::views::iota(0ULL, n)) {
        UINT dat = random.int32() % 12;
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
    ParamGate g1 = gate::PPauliRotation(PauliOperator(data1), 1., controls);
    ParamGate g2 = gate::PPauliRotation(PauliOperator(data2), 1., {});
    test_gate(g1, g2, n, control_mask, param);
}

TEST(ParamGateTest, Control) {
    UINT n = 10;
    for ([[maybe_unused]] UINT _ : std::views::iota(0, 10)) {
        test_param_rotation_control(gate::PRX, n);
        test_param_rotation_control(gate::PRY, n);
        test_param_rotation_control(gate::PRZ, n);
        test_ppauli_control(n);
    }
}
