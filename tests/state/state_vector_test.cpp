#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <state/state_vector.hpp>
#include <util/utility.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using CComplex = std::complex<double>;

using namespace scaluq;

const double eps = 1e-12;

TEST(StateVectorTest, HaarRandomStateNorm) {
    const int n_tries = 20;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVector::Haar_random_state(n);
        ASSERT_NEAR(state.get_squared_norm(), 1., eps);
    }
}

TEST(StateVectorTest, OperationAtIndex) {
    auto state = StateVector::Haar_random_state(10);
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        state.set_amplitude_at(i, 1);
        ASSERT_NEAR(state.get_amplitude_at(i).real(), 1, eps);
        ASSERT_NEAR(state.get_amplitude_at(i).imag(), 0., eps);
    }
}

TEST(StateVectorTest, CopyState) {
    const int n = 5;
    const auto state = StateVector::Haar_random_state(n);
    StateVector state_cp = state.copy();
    auto vec1 = state.get_amplitudes();
    auto vec2 = state_cp.get_amplitudes();
    ASSERT_EQ(vec1, vec2);
}

TEST(StateVectorTest, ZeroNormState) {
    const std::uint64_t n = 5;

    StateVector state(StateVector::Haar_random_state(n));
    state.set_zero_norm_state();
    auto state_cp = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        ASSERT_EQ((CComplex)state_cp[i], CComplex(0, 0));
    }
}

TEST(StateVectorTest, ComputationalBasisState) {
    const std::uint64_t n = 5;

    StateVector state(StateVector::Haar_random_state(n));
    state.set_computational_basis(31);
    auto state_cp = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        if (i == 31) {
            ASSERT_EQ((CComplex)state_cp[i], CComplex(1, 0));
        } else {
            ASSERT_EQ((CComplex)state_cp[i], CComplex(0, 0));
        }
    }
}

TEST(StateVectorTest, HaarRandomStateSameSeed) {
    const std::uint64_t n = 10, m = 5;
    for (std::uint64_t i = 0; i < m; ++i) {
        StateVector state1(StateVector::Haar_random_state(n, i)),
            state2(StateVector::Haar_random_state(n, i));
        ASSERT_TRUE(same_state(state1, state2));
    }
}

TEST(StateVectorTest, HaarRandomStateWithoutSeed) {
    const std::uint64_t n = 10, m = 5;
    for (std::uint64_t i = 0; i < m; ++i) {
        StateVector state1(StateVector::Haar_random_state(n)),
            state2(StateVector::Haar_random_state(n));
        ASSERT_FALSE(same_state(state1, state2));
    }
}

TEST(StateVectorTest, AddState) {
    const std::uint64_t n = 10;
    StateVector state1(StateVector::Haar_random_state(n));
    StateVector state2(StateVector::Haar_random_state(n));
    auto vec1 = state1.get_amplitudes();
    auto vec2 = state2.get_amplitudes();
    state1.add_state_vector(state2);
    auto new_vec = state1.get_amplitudes();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
        CComplex res = new_vec[i], val = (CComplex)vec1[i] + (CComplex)vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps);
        ASSERT_NEAR(res.imag(), val.imag(), eps);
    }
}

TEST(StateVectorTest, AddStateWithCoef) {
    const CComplex coef(2.5, 1.3);
    const std::uint64_t n = 10;
    StateVector state1(StateVector::Haar_random_state(n));
    StateVector state2(StateVector::Haar_random_state(n));
    auto vec1 = state1.get_amplitudes();
    auto vec2 = state2.get_amplitudes();

    state1.add_state_vector_with_coef(coef, state2);
    auto new_vec = state1.get_amplitudes();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
        CComplex res = new_vec[i], val = (CComplex)vec1[i] + coef * (CComplex)vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps);
        ASSERT_NEAR(res.imag(), val.imag(), eps);
    }
}

TEST(StateVectorTest, MultiplyCoef) {
    const std::uint64_t n = 10;
    const CComplex coef(0.5, 0.2);

    StateVector state(StateVector::Haar_random_state(n));
    auto vec = state.get_amplitudes();
    state.multiply_coef(coef);
    auto new_vec = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        CComplex res = new_vec[i], val = coef * (CComplex)vec[i];
        ASSERT_NEAR(res.real(), val.real(), eps);
        ASSERT_NEAR(res.imag(), val.imag(), eps);
    }
}

TEST(StateVectorTest, GetZeroProbability) {
    const std::uint64_t n = 10;
    StateVector state(n);
    state.set_computational_basis(1);
    for (std::uint64_t i = 2; i <= 10; ++i) {
        StateVector tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_vector_with_coef(std::sqrt(i), tmp_state);
    }
    state.normalize();
    ASSERT_NEAR(state.get_zero_probability(0), 30.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(1), 27.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(2), 33.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(3), 28.0 / 55.0, eps);
}

TEST(StateVectorTest, EntropyCalculation) {
    const std::uint64_t n = 6;
    const std::uint64_t dim = 1ULL << n;
    const std::uint64_t max_repeat = 10;

    StateVector state(n);
    for (std::uint64_t rep = 0; rep < max_repeat; ++rep) {
        state = StateVector::Haar_random_state(n);
        auto state_cp = state.get_amplitudes();
        ASSERT_NEAR(state.get_squared_norm(), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state[i] = (CComplex)state_cp[i];

        for (std::uint64_t target = 0; target < n; ++target) {
            double ent = 0;
            for (std::uint64_t ind = 0; ind < dim; ++ind) {
                CComplex z = test_state[ind];
                double prob = z.real() * z.real() + z.imag() * z.imag();
                if (prob > eps) ent += -prob * log2(prob);
            }
            ASSERT_NEAR(ent, state.get_entropy(), eps);
        }
    }
}

TEST(StateVectorTest, GetMarginalProbability) {
    const std::uint64_t n = 2;
    const std::uint64_t dim = 1 << n;
    StateVector state(StateVector::Haar_random_state(n));
    auto state_cp = state.get_amplitudes();
    std::vector<double> probs;
    for (std::uint64_t i = 0; i < dim; ++i) {
        probs.push_back(internal::squared_norm(state_cp[i]));
    }
    ASSERT_NEAR(state.get_marginal_probability({0, 0}), probs[0], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 0}), probs[1], eps);
    ASSERT_NEAR(state.get_marginal_probability({0, 1}), probs[2], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 1}), probs[3], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({0, StateVector::UNMEASURED}), probs[0] + probs[2], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({1, StateVector::UNMEASURED}), probs[1] + probs[3], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({StateVector::UNMEASURED, 0}), probs[0] + probs[1], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({StateVector::UNMEASURED, 1}), probs[2] + probs[3], eps);
    ASSERT_NEAR(state.get_marginal_probability({StateVector::UNMEASURED, StateVector::UNMEASURED}),
                1.,
                eps);
}

TEST(StateVectorTest, SamplingSuperpositionState) {
    const std::uint64_t n = 10;
    const std::uint64_t nshot = 65536;
    const std::uint64_t test_count = 10;
    std::uint64_t pass_count = 0;
    for (std::uint64_t test_i = 0; test_i < test_count; test_i++) {
        StateVector state(n);
        state.set_computational_basis(0);
        for (std::uint64_t i = 1; i <= 4; ++i) {
            StateVector tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_vector_with_coef(1 << i, tmp_state);
        }
        state.normalize();
        std::vector<size_t> res = state.sampling(nshot);

        std::array<std::uint64_t, 5> cnt = {};
        for (std::uint64_t i = 0; i < nshot; ++i) {
            ASSERT_GE(res[i], 0);
            ASSERT_LE(res[i], 4);
            cnt[res[i]] += 1;
        }
        bool pass = true;
        for (std::uint64_t i = 0; i < 4; i++) {
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

TEST(StateVectorTest, SamplingComputationalBasis) {
    const std::uint64_t n = 10;
    const std::uint64_t nshot = 1024;
    StateVector state(n);
    state.set_computational_basis(100);
    auto res = state.sampling(nshot);
    for (std::uint64_t i = 0; i < nshot; ++i) {
        ASSERT_TRUE(res[i] == 100);
    }
}
