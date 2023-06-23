#include <gtest/gtest.h>

#include <cpusim/state_vector_cpu.hpp>

const double eps = 1e-12;

TEST(StateTest, HaarRandomStateNorm) {
    const int n_tries = 20;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVectorCpu::Haar_random_state(n);
        ASSERT_NEAR(state.compute_squared_norm(), 1., eps);
    }
}
