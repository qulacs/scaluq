#include <gtest/gtest.h>

#include <gate/gate_factory.hpp>
#include <gate/pgate_factory.hpp>
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
        const Gate gate = factory_fixed(target, pcoef * param);
        const PGate pgate = factory_parametric(target, pcoef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.amplitudes();
        auto state_cp_amp = state_cp.amplitudes();

        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }

        PGate pgate_inv = pgate->get_inverse();
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
        Gate gate = PauliRotation(pauli, pcoef * param);
        PGate pgate = PPauliRotation(pauli, pcoef);
        gate->update_quantum_state(state);
        pgate->update_quantum_state(state_cp, param);
        auto state_amp = state.amplitudes();
        auto state_cp_amp = state_cp.amplitudes();
        // check if the state is updated correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs(state_cp_amp[i] - state_amp[i]), 0, eps);
        }
        PGate pgate_inv = pgate->get_inverse();
        pgate_inv->update_quantum_state(state, param);
        state_amp = state.amplitudes();
        auto state_bef_amp = state_bef.amplitudes();
        // check if the state is restored correctly
        for (UINT i = 0; i < dim; i++) {
            ASSERT_NEAR(Kokkos::abs((state_bef_amp[i] - state_amp[i])), 0, eps);
        }
    }
}

TEST(PGateTest, ApplyPRXGate) { test_apply_parametric_single_pauli_rotation(5, &RX, &PRX); }
TEST(PGateTest, ApplyPRYGate) { test_apply_parametric_single_pauli_rotation(5, &RX, &PRX); }
TEST(PGateTest, ApplyPRZGate) { test_apply_parametric_single_pauli_rotation(5, &RX, &PRX); }
TEST(PGateTest, ApplyPPauliRotationGate) { test_apply_parametric_multi_pauli_rotation(5); }
