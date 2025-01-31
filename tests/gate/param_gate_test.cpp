#include <gtest/gtest.h>

#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>

template <std::floating_point Fp,
          ExecutionSpace Sp,
          typename FactoryFixed,
          typename FactoryParametric>
void test_apply_parametric_single_pauli_rotation(std::uint64_t n_qubits,
                                                 FactoryFixed factory_fixed,
                                                 FactoryParametric factory_parametric) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();

        const std::uint64_t target = random.int32() % n_qubits;
        const Fp param = std::numbers::pi * random.uniform();
        const Fp param_coef = random.uniform() * 2 - 1;
        const Gate<Fp, Sp> gate = factory_fixed(target, param_coef * param, {});
        const ParamGate<Fp, Sp> pgate = factory_parametric(target, param_coef, {});
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();

        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps<Fp>);
        }

        ParamGate<Fp, Sp> pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_bef_amp[i] - state_amp[i]), 0, eps<Fp>);
        }
    }
}

FLOAT_AND_SPACE(Fp, Sp)
void test_apply_parametric_multi_pauli_rotation(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();
        const Fp param = std::numbers::pi * random.uniform();
        const Fp param_coef = random.uniform() * 2 - 1;
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        PauliOperator<Fp, Sp> pauli(target_vec, pauli_id_vec, 1.0);
        Gate gate = gate::PauliRotation<Fp, Sp>(pauli, param_coef * param);
        ParamGate pgate = gate::ParamPauliRotation<Fp, Sp>(pauli, param_coef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();
        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps<Fp>);
        }
        ParamGate<Fp, Sp> pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs((state_bef_amp[i] - state_amp[i])), 0, eps<Fp>);
        }
    }
}

TEST(ParamGateTest, ApplyParamRXGate) {
    test_apply_parametric_single_pauli_rotation<double, DefaultSpace>(
        5, &gate::RX<double, DefaultSpace>, &gate::ParamRX<double, DefaultSpace>);
    test_apply_parametric_single_pauli_rotation<float, CPUSpace>(
        5, &gate::RX<float, CPUSpace>, &gate::ParamRX<float, CPUSpace>);
}
TEST(ParamGateTest, ApplyParamRYGate) {
    test_apply_parametric_single_pauli_rotation<double, DefaultSpace>(
        5, &gate::RY<double, DefaultSpace>, &gate::ParamRY<double, DefaultSpace>);
}
TEST(ParamGateTest, ApplyParamRZGate) {
    test_apply_parametric_single_pauli_rotation<double, DefaultSpace>(
        5, &gate::RZ<double, DefaultSpace>, &gate::ParamRZ<double, DefaultSpace>);
}
TEST(ParamGateTest, ApplyParamPauliRotationGate) {
    test_apply_parametric_multi_pauli_rotation<double, DefaultSpace>(5);
    test_apply_parametric_multi_pauli_rotation<float, CPUSpace>(5);
}

FLOAT_AND_SPACE(Fp, Sp)
void TestParamProbablisticGate(std::uint64_t n_qubits) {
    auto probgate =
        gate::ParamProbablistic<Fp, Sp>({.1, .9}, {gate::ParamRX<Fp, Sp>(0), gate::I<Fp, Sp>()});
    std::uint64_t x_cnt = 0, i_cnt = 0;
    StateVector<Fp, Sp> state(1);
    for ([[maybe_unused]] auto _ : std::views::iota(0, 100)) {
        std::uint64_t before = state.sampling(1)[0];
        probgate->update_quantum_state(state, std::numbers::pi);
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

TEST(ParamGateTest, ApplyParamProbablisticGate) {
    TestParamProbablisticGate<double, DefaultSpace>(5);
    TestParamProbablisticGate<float, CPUSpace>(5);
}

FLOAT_AND_SPACE(Fp, Sp)
void test_gate(ParamGate<Fp, Sp> gate_control,
               ParamGate<Fp, Sp> gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask,
               Fp param) {
    StateVector state = StateVector<Fp, Sp>::Haar_random_state(n_qubits);
    auto amplitudes = state.get_amplitudes();
    StateVector<Fp, Sp> state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<Complex<Fp>> amplitudes_controlled(state_controlled.dim());
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
            eps<Fp>);
    }
}

template <std::floating_point Fp, ExecutionSpace Sp, typename Factory>
void test_param_rotation_control(Factory factory, std::uint64_t n) {
    Random random;
    std::vector<std::uint64_t> shuffled = random.permutation(n);
    std::uint64_t target = shuffled[0];
    std::uint64_t num_control = random.int32() % n;
    std::vector<std::uint64_t> controls(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[1 + i];
    }
    std::uint64_t control_mask = 0ULL;
    for (std::uint64_t c : controls) control_mask |= 1ULL << c;
    Fp param = random.uniform() * std::numbers::pi * 2;
    ParamGate<Fp, Sp> g1 = factory(target, 1., controls);
    ParamGate<Fp, Sp> g2 =
        factory(target - std::popcount(control_mask & ((1ULL << target) - 1)), 1., {});
    test_gate(g1, g2, n, control_mask, param);
}

FLOAT_AND_SPACE(Fp, Sp)
void test_ppauli_control(std::uint64_t n) {
    typename PauliOperator<Fp, Sp>::Data data1, data2;
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
    double param = random.uniform() * std::numbers::pi * 2;
    ParamGate g1 = gate::ParamPauliRotation<Fp, Sp>(PauliOperator<Fp, Sp>(data1), 1., controls);
    ParamGate g2 = gate::ParamPauliRotation<Fp, Sp>(PauliOperator<Fp, Sp>(data2), 1., {});
    test_gate<Fp, Sp>(g1, g2, n, control_mask, param);
}

TEST(ParamGateTest, Control) {
    test_param_rotation_control<double, DefaultSpace>(&gate::ParamRX<double, DefaultSpace>, 6);
    test_param_rotation_control<float, CPUSpace>(&gate::ParamRX<float, CPUSpace>, 6);
    test_param_rotation_control<double, DefaultSpace>(&gate::ParamRY<double, DefaultSpace>, 6);
    test_param_rotation_control<float, CPUSpace>(&gate::ParamRY<float, CPUSpace>, 6);
    test_param_rotation_control<double, DefaultSpace>(&gate::ParamRZ<double, DefaultSpace>, 6);
    test_param_rotation_control<float, CPUSpace>(&gate::ParamRZ<float, CPUSpace>, 6);
    test_ppauli_control<double, DefaultSpace>(6);
    test_ppauli_control<float, CPUSpace>(6);
}
