#include <gtest/gtest.h>

#include <state/state_vector.hpp>

#include "test_environment.hpp"

namespace qulacs {
const double eps = 1e-12;

TEST(DummyTest, Dummy) {
    const int n_tries = 20;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVector(n);
        ASSERT_EQ(state.amplitudes_raw().size(), 1ULL << n);
    }
}
}  // namespace qulacs
