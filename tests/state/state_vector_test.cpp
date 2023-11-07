#include <gtest/gtest.h>

#include <state/state_vector.hpp>

#include "../test_environment.hpp"

namespace qulacs {
const double eps = 1e-12;

TEST(StateVectorTest, HaarRandomStateNorm) {
    const int n_tries = 20;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVector::Haar_random_state(n);
        ASSERT_NEAR(state.compute_squared_norm(), 1., eps);
    }
}

TEST(StateVectorTest, GetZeroProbability) {
    const UINT n = 10;

    StateVector state(n);
    state.set_computational_basis(1);
    for (UINT i = 2; i <= 10; ++i) {
        StateVector tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_with_coef(std::sqrt(i), tmp_state);
    }
    state.normalize();
    ASSERT_NEAR(state.get_zero_probability(0), 30.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(1), 27.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(2), 33.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(3), 28.0 / 55.0, eps);
}

}  // namespace qulacs
