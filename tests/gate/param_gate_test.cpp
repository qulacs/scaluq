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
        const double pcoef = random.uniform() * 2 - 1;
        const Gate gate = factory_fixed(target, pcoef * param, {});
        const ParamGate pgate = factory_parametric(target, pcoef, {});
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.amplitudes();
        auto state_cp_amp = state_cp.amplitudes();

        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }

        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.amplitudes();
        auto state_bef_amp = state_bef.amplitudes();
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
        const double pcoef = random.uniform() * 2 - 1;
        std::vector<std::uint64_t> target_vec, pauli_id_vec;
        for (std::uint64_t target = 0; target < n_qubits; target++) {
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
        for (std::uint64_t i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }
        ParamGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.amplitudes();
        auto state_bef_amp = state_bef.amplitudes();
        // check if the state is restored correctly
        for (std::uint64_t i = 0; i < dim; i++) {
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
