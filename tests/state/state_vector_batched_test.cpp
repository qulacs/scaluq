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
    auto states = StateVectorBatched::Haar_random_states(batch_size, n_qubits);
    StateVector zeronorm(n_qubits);
    zeronorm.set_zero_norm_state();
    states.set_state_vector_at_batch_id(3, zeronorm);
    std::cout << states.to_string() << std::endl;
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
