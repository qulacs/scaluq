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
    const int batch_size = 10, n_qubits = 10;
    const auto states = StateVectorBatched::Haar_random_state(batch_size, n_qubits);
    auto norms = states.get_squared_norm();
    for (auto x : norms) ASSERT_NEAR(x, 1., eps);
}

TEST(StateVectorBatchedTest, OperationAtIndex) {
    const int batch_size = 10, n_qubits = 3;
    auto states = StateVectorBatched::Haar_random_state(batch_size, n_qubits);
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
    for (UINT b = 0; b < batch_size; ++b) {
        for (UINT i = 0; i < states.dim(); ++i) {
            states.set_amplitude_at_index(b, i, Complex(b, i));
        }
    }
    for (UINT b = 0; b < batch_size; ++b) {
        for (UINT i = 0; i < states.dim(); ++i) {
            auto c = states.get_amplitude_at_index(b, i);
            ASSERT_NEAR(c.real(), b, eps);
            ASSERT_NEAR(c.imag(), i, eps);
        }
    }
    auto states_cpu = states.amplitudes();
    for (UINT b = 0; b < batch_size; ++b) {
        for (UINT i = 0; i < states.dim(); ++i) {
            ASSERT_NEAR(states_cpu[b][i].real(), b, eps);
            ASSERT_NEAR(states_cpu[b][i].imag(), i, eps);
        }
    }
}
