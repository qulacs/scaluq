#include <gtest/gtest.h>

#include <Eigen/Core>
#include <state/state_vector.hpp>

#include "../test_environment.hpp"
#include "util/utility.hpp"

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

TEST(StateVectorTest, GetMarginalProbability) {
    const UINT n = 2;
    const UINT dim = 1 << n;
    StateVector state(n);
    state = StateVector::Haar_random_state(n);
    std::vector<double> probs;
    for (UINT i = 0; i < dim; ++i) {
        probs.push_back(std::norm(state[i]));
    }
    ASSERT_NEAR(state.get_marginal_probability({0, 0}), probs[0], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 0}), probs[1], eps);
    ASSERT_NEAR(state.get_marginal_probability({0, 1}), probs[2], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 1}), probs[3], eps);
    ASSERT_NEAR(state.get_marginal_probability({0, 2}), probs[0] + probs[2], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 2}), probs[1] + probs[3], eps);
    ASSERT_NEAR(state.get_marginal_probability({2, 0}), probs[0] + probs[1], eps);
    ASSERT_NEAR(state.get_marginal_probability({2, 1}), probs[2] + probs[3], eps);
    ASSERT_NEAR(state.get_marginal_probability({2, 2}), 1., eps);
}

TEST(StateVectorTest, EntropyCalculation) {
    const UINT n = 6;
    const UINT dim = 1ULL << n;
    const UINT max_repeat = 10;

    StateVector state(n);
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        state = StateVector::Haar_random_state(n);
        ASSERT_NEAR(state.compute_squared_norm(), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (UINT i = 0; i < dim; ++i) test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            double ent = 0;
            for (UINT ind = 0; ind < dim; ++ind) {
                double prob = std::norm(test_state[ind]);
                if (prob > eps) ent += -prob * log(prob);
            }
            ASSERT_NEAR(ent, state.get_entropy(), eps);
        }
    }
}

TEST(StateVectorTest, SamplingSuperpositionState) {
    const UINT n = 10;
    const UINT nshot = 1024;
    const UINT test_count = 10;
    UINT pass_count = 0;
    for (UINT test_i = 0; test_i < test_count; test_i++) {
        StateVector state(n);
        state.set_computational_basis(0);
        for (UINT i = 1; i <= 4; ++i) {
            StateVector tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_with_coef(1 << i, tmp_state);
        }
        state.normalize();
        auto res = state.sampling(nshot);

        std::array<UINT, 5> cnt = {};
        for (UINT i = 0; i < nshot; ++i) {
            ASSERT_GE(res[i], 0);
            ASSERT_LE(res[i], 4);
            cnt[res[i]] += 1;
        }
        bool pass = true;
        for (UINT i = 0; i < 4; i++) {
            std::string err_message = _CHECK_GT(cnt[i + 1], cnt[i]);
            if (err_message != "") {
                pass = false;
                std::cerr << err_message;
            }
        }
        if (pass) pass_count++;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TEST(StateVectorTest, ExecutionSpaceTest) {
    StateVector tmp;
#ifdef __CUDA_ARCH__
    ASSERT_EQ(tmp.get_device_name(), "gpu");
#else
    ASSERT_EQ(tmp.get_device_name(), "cpu");
#endif
}

}  // namespace qulacs
