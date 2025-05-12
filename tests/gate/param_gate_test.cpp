#include <gtest/gtest.h>

#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class ParamGateTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(ParamGateTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space, typename FactoryFixed, typename FactoryParametric>
void test_apply_parametric_single_pauli_rotation(std::uint64_t n_qubits,
                                                 FactoryFixed factory_fixed,
                                                 FactoryParametric factory_parametric) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        auto state = StateVector<Prec, Space>::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();

        const std::uint64_t target = random.int32() % n_qubits;
        const double param = std::numbers::pi * random.uniform();
        const double param_coef = random.uniform() * 2 - 1;
        const Gate<Prec, Space> gate = factory_fixed(target, param_coef * param, {}, {});
        const ParamGate<Prec, Space> pgate = factory_parametric(target, param_coef, {}, {});
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();

        for (std::uint64_t i = 0; i < dim; i++) {
            check_near<Prec>(state_amp[i], state_cp_amp[i]);
        }

        ParamGate<Prec, Space> pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near<Prec>(state_amp[i], state_bef_amp[i]);
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
void test_apply_parametric_multi_pauli_rotation(std::uint64_t n_qubits) {
    const std::uint64_t dim = 1ULL << n_qubits;
    Random random;

    for (int repeat = 0; repeat < 10; repeat++) {
        StateVector state = StateVector<Prec, Space>::Haar_random_state(n_qubits);
        auto state_cp = state.copy();
        auto state_bef = state.copy();
        const double param = std::numbers::pi * random.uniform();
        const double param_coef = random.uniform() * 2 - 1;
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
            target_vec.emplace_back(target);
            pauli_id_vec.emplace_back(random.int64() % 4);
        }

        PauliOperator<Prec, Space> pauli(target_vec, pauli_id_vec, 1.0);
        Gate gate = gate::PauliRotation<Prec, Space>(pauli, param_coef * param);
        ParamGate pgate = gate::ParamPauliRotation<Prec, Space>(pauli, param_coef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.get_amplitudes();
        auto state_cp_amp = state_cp.get_amplitudes();
        // check if the state is updated correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near<Prec>(state_amp[i], state_cp_amp[i]);
        }
        ParamGate<Prec, Space> pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.get_amplitudes();
        auto state_bef_amp = state_bef.get_amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
            check_near<Prec>(state_amp[i], state_bef_amp[i]);
        }
    }
}

TYPED_TEST(ParamGateTest, ApplyParamRXGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    test_apply_parametric_single_pauli_rotation<Prec, Space>(
        5, &gate::RX<Prec, Space>, &gate::ParamRX<Prec, Space>);
}
TYPED_TEST(ParamGateTest, ApplyParamRYGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    test_apply_parametric_single_pauli_rotation<Prec, Space>(
        5, &gate::RX<Prec, Space>, &gate::ParamRX<Prec, Space>);
}
TYPED_TEST(ParamGateTest, ApplyParamRZGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    test_apply_parametric_single_pauli_rotation<Prec, Space>(
        5, &gate::RX<Prec, Space>, &gate::ParamRX<Prec, Space>);
}
TYPED_TEST(ParamGateTest, ApplyParamPauliRotationGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    test_apply_parametric_multi_pauli_rotation<Prec, Space>(5);
}

TYPED_TEST(ParamGateTest, ApplyParamProbabilisticGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    auto probgate = gate::ParamProbabilistic<Prec, Space>(
        {.1, .9}, {gate::ParamRX<Prec, Space>(0), gate::I<Prec, Space>()});
    std::uint64_t x_cnt = 0, i_cnt = 0;
    StateVector<Prec, Space> state(1);
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
    // These test is probabilistic, but pass at least 99.99% cases.
    ASSERT_GT(x_cnt, 0);
    ASSERT_GT(i_cnt, 0);
    ASSERT_LT(x_cnt, i_cnt);
}

template <Precision Prec, ExecutionSpace Space>
void test_gate(ParamGate<Prec, Space> gate_control,
               ParamGate<Prec, Space> gate_simple,
               std::uint64_t n_qubits,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               double param) {
    StateVector state = StateVector<Prec, Space>::Haar_random_state(n_qubits);
    auto amplitudes = state.get_amplitudes();
    StateVector<Prec, Space> state_controlled(n_qubits - std::popcount(control_mask));
    std::vector<StdComplex> amplitudes_controlled(state_controlled.dim());
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        amplitudes_controlled[i] =
            amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                       control_value_mask];
    }
    state_controlled.load(amplitudes_controlled);
    gate_control->update_quantum_state(state, param);
    gate_simple->update_quantum_state(state_controlled, param);
    amplitudes = state.get_amplitudes();
    amplitudes_controlled = state_controlled.get_amplitudes();
    for (std::uint64_t i : std::views::iota(0ULL, state_controlled.dim())) {
        check_near<Prec>(amplitudes[internal::insert_zero_at_mask_positions(i, control_mask) |
                                    control_value_mask],
                         amplitudes_controlled[i]);
    }
}

template <Precision Prec, ExecutionSpace Space, typename Factory>
void test_param_rotation_control(Factory factory, std::uint64_t n) {
    Random random;
    std::vector<std::uint64_t> shuffled = random.permutation(n);
    std::uint64_t target = shuffled[0];
    std::uint64_t num_control = random.int32() % (n);
    std::vector<std::uint64_t> controls(num_control), control_values(num_control);
    for (std::uint64_t i : std::views::iota(0ULL, num_control)) {
        controls[i] = shuffled[1 + i];
        control_values[i] = random.int32() & 1;
    }
    std::uint64_t control_mask = 0ULL, control_value_mask = 0ULL;
    for (std::uint64_t i = 0; i < num_control; ++i) {
        control_mask |= 1ULL << controls[i];
        control_value_mask |= control_values[i] << controls[i];
    }
    double param = random.uniform() * std::numbers::pi * 2;
    ParamGate<Prec, Space> g1 = factory(target, 1., controls, control_values);
    ParamGate<Prec, Space> g2 =
        factory(target - std::popcount(control_mask & ((1ULL << target) - 1)), 1., {}, {});
    test_gate(g1, g2, n, control_mask, control_value_mask, param);
}

template <Precision Prec, ExecutionSpace Space>
void test_ppauli_control(std::uint64_t n) {
    typename PauliOperator<Prec, Space>::Data data1, data2;
    std::vector<std::uint64_t> controls, control_values;
    std::uint64_t control_mask = 0, control_value_mask = 0;
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
            bool v = random.int64() & 1;
            control_values.push_back(v);
            control_value_mask |= v << i;
            num_control++;
        }
    }
    double param = random.uniform() * std::numbers::pi * 2;
    ParamGate g1 =
        gate::ParamPauliRotation(PauliOperator<Prec, Space>(data1), 1., controls, control_values);
    ParamGate g2 = gate::ParamPauliRotation(PauliOperator<Prec, Space>(data2), 1., {}, {});
    test_gate(g1, g2, n, control_mask, control_value_mask, param);
}

TYPED_TEST(ParamGateTest, Control) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 10;
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, 10)) {
        test_param_rotation_control<Prec, Space>(gate::ParamRX<Prec, Space>, n);
        test_param_rotation_control<Prec, Space>(gate::ParamRY<Prec, Space>, n);
        test_param_rotation_control<Prec, Space>(gate::ParamRZ<Prec, Space>, n);
        test_ppauli_control<Prec, Space>(n);
    }
}
