#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <state/state_vector_batched.hpp>
#include <util/utility.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using CComplex = std::complex<double>;

using namespace scaluq;

const double eps = 1e-12;

TEST(StateVectorBatchedTest, HaarRandomStateNorm) {
    const UINT batch_size = 10, n_qubits = 3;
    const auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits);
    auto norms = states.get_squared_norm();
    for (auto x : norms) ASSERT_NEAR(x, 1., eps);
}

TEST(StateVectorBatchedTest, LoadAndAmplitues) {
    const UINT batch_size = 4, n_qubits = 3;
    const UINT dim = 1 << n_qubits;
    std::vector states_h(batch_size, std::vector<Complex>(dim));
    for (UINT b = 0; b < batch_size; ++b) {
        for (UINT i = 0; i < dim; ++i) {
            states_h[b][i] = b * dim + i;
        }
    }
    StateVectorBatched states(batch_size, n_qubits);

    states.load(states_h);
    auto amps = states.amplitudes();
    for (UINT b = 0; b < batch_size; ++b) {
        for (UINT i = 0; i < dim; ++i) {
            ASSERT_EQ(amps[b][i].real(), b * dim + i);
        }
    }
}

TEST(StateVectorBatchedTest, OperateState) {
    const UINT batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);
    auto states_add = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 1);
    const Complex coef(2.1, 3.5);

    auto states_cp = states.copy();
    for (UINT b = 0; b < batch_size; ++b) {
        ASSERT_TRUE(same_state(states.get_state_vector(b), states_cp.get_state_vector(b)));
    }

    states.add_state_vector(states_add);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector(b);
        v.add_state_vector(states_add.get_state_vector(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector(b)));
    }

    states_cp = states.copy();
    states.add_state_vector_with_coef(coef, states_add);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector(b);
        v.add_state_vector_with_coef(coef, states_add.get_state_vector(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector(b)));
    }

    states_cp = states.copy();
    states.multiply_coef(coef);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector(b);
        v.multiply_coef(coef);
        ASSERT_TRUE(same_state(v, states.get_state_vector(b)));
    }
}

TEST(StateVectorBatchedTest, ZeroProbs) {
    const UINT batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);

    for (UINT i = 0; i < n_qubits; ++i) {
        auto zero_probs = states.get_zero_probability(i);
        for (UINT b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector(b);
            ASSERT_NEAR(zero_probs[b], state.get_zero_probability(i), eps);
        }
    }
}

TEST(StateVectorBatchedTest, MarginalProbs) {
    const UINT batch_size = 4, n_qubits = 5;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);

    Random rd(0);
    for (UINT i = 0; i < 10; ++i) {
        std::vector<UINT> targets;
        for (UINT j = 0; j < n_qubits; ++j) {
            targets.push_back(rd.int32() % 3);
        }
        auto mg_probs = states.get_marginal_probability(targets);
        for (UINT b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector(b);
            ASSERT_NEAR(mg_probs[b], state.get_marginal_probability(targets), eps);
        }
    }
}

TEST(StateVectorBatchedTest, Entropy) {
    const UINT batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);

    auto entropies = states.get_entropy();
    for (UINT b = 0; b < batch_size; ++b) {
        auto state = states.get_state_vector(b);
        ASSERT_NEAR(entropies[b], state.get_entropy(), eps);
    }
}
