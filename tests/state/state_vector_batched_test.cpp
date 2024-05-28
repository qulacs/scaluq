#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <state/state_vector.hpp>
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

TEST(StateVectorBatchedTest, OperationAtIndex) {
    const UINT batch_size = 10, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits);
    for (UINT i = 0; i < states.dim(); ++i) {
        states.set_amplitude_at_index(i, Complex(i, 0));
    }
    for (UINT i = 0; i < states.dim(); ++i) {
        auto cs = states.get_amplitude_at_index(i);
        for (auto c : cs) {
            ASSERT_NEAR(c.real(), i, eps);
            ASSERT_NEAR(c.imag(), 0., eps);
        }
    }
}

TEST(StateVectorBatchedTest, ToString) {
    const UINT batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);
    StateVector zeronorm(n_qubits);
    zeronorm.set_zero_norm_state();
    states.set_state_vector_at_batch_id(3, zeronorm);
    std::string expected_output = 
    " *** Quantum States ***\n"
    " * Qubit Count : 3\n"
    " * Dimension   : 8\n"
    "--------------------\n"
    " * Batch_id    : 0\n"
    " * State vector : \n"
    "(0.0198679,-0.472614)\n"
    "(-0.571834,0.263848)\n"
    "(0.314844,-0.0148995)\n"
    "(-0.229146,0.0968181)\n"
    "(-0.0295159,-0.147023)\n"
    "(0.157666,-0.280552)\n"
    "(0.0500783,-0.299107)\n"
    "(0.0058146,-0.0184111)\n"
    "--------------------\n"
    " * Batch_id    : 1\n"
    " * State vector : \n"
    "(-0.297257,0.282742)\n"
    "(-0.108597,-0.106435)\n"
    "(-0.373668,0.368799)\n"
    "(0.272888,-0.226803)\n"
    "(-0.481157,-0.166105)\n"
    "(0.0478232,0.101383)\n"
    "(-0.135995,-0.0720401)\n"
    "(-0.327676,0.0655819)\n"
    "--------------------\n"
    " * Batch_id    : 2\n"
    " * State vector : \n"
    "(0.443181,-0.113553)\n"
    "(-0.0726955,-0.194018)\n"
    "(0.429067,0.0895584)\n"
    "(0.110552,0.22536)\n"
    "(-0.269319,-0.123394)\n"
    "(-0.243684,-0.0950806)\n"
    "(-0.342079,-0.112199)\n"
    "(-0.297306,0.344185)\n"
    "--------------------\n"
    " * Batch_id    : 3\n"
    " * State vector : \n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n"
    "(0,0)\n";
    EXPECT_EQ(states.to_string(), expected_output);
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

TEST(StateVectorBatchedTest, AddState) {
    const UINT batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 0);
    auto states_add = StateVectorBatched::Haar_random_states(batch_size, n_qubits, 1);
    const Complex coef(2.1, 3.5);

    auto states_cp = states.copy();
    for (UINT b = 0; b < batch_size; ++b) {
        ASSERT_TRUE(same_state(states.get_state_vector_at_batch_id(b), states_cp.get_state_vector_at_batch_id(b)));
    }

    states.add_state_vector(states_add);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at_batch_id(b);
        v.add_state_vector(states_add.get_state_vector_at_batch_id(b));
        std::cout << v << std::endl;
        std::cout << states_cp.get_state_vector_at_batch_id(b) << std::endl;
        ASSERT_TRUE(same_state(v, states.get_state_vector_at_batch_id(b)));
    }
    
    states_cp = states.copy();
    states.add_state_vector_with_coef(coef, states_add);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at_batch_id(b);
        v.add_state_vector_with_coef(coef, states_add.get_state_vector_at_batch_id(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector_at_batch_id(b)));
    }

    states_cp = states.copy();
    states.multiply_coef(coef);
    for (UINT b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at_batch_id(b);
        v.multiply_coef(coef);
        ASSERT_TRUE(same_state(v, states.get_state_vector_at_batch_id(b)));
    }
}


