#include <gtest/gtest.h>

#include <state/state_vector.hpp>

#include "../test_environment.hpp"

const double eps = 1e-12;

TEST(StateVectorTest, HaarRandomStateNorm) {
    const int n_tries = 20;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVector::Haar_random_state(n);
        ASSERT_NEAR(state.compute_squared_norm(), 1., eps);
    }
}
