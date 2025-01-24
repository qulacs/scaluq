#include <gtest/gtest.h>

#include <scaluq/state/state_vector.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>
#define EXECUTE_TEST(Name, arg)                \
    TEST(StateVectorTest, Name) {              \
        Test##Name<double, DefaultSpace>(arg); \
        Test##Name<double, CPUSpace>(arg);     \
        Test##Name<float, DefaultSpace>(arg);  \
        Test##Name<float, CPUSpace>(arg);      \
    }

FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateNorm(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    ASSERT_NEAR(state.get_squared_norm(), 1.0, eps<Fp>);
}
EXECUTE_TEST(HaarRandomStateNorm, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestOperationAtIndex(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        state.set_amplitude_at(i, 1);
        ASSERT_NEAR(state.get_amplitude_at(i).real(), 1.0, eps<Fp>);
        ASSERT_NEAR(state.get_amplitude_at(i).imag(), 0.0, eps<Fp>);
    }
}
EXECUTE_TEST(OperationAtIndex, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestCopyState(std::uint32_t n) {
    const auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    StateVector<Fp, Sp> state_cp = state.copy();
    auto vec1 = state.get_amplitudes();
    auto vec2 = state_cp.get_amplitudes();
    ASSERT_EQ(vec1, vec2);
}
EXECUTE_TEST(CopyState, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestZeroNormState(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    state.set_zero_norm_state();
    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(0, 0));
    }
}
EXECUTE_TEST(ZeroNormState, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestComputationalBasisState(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    state.set_computational_basis(31);
    auto state_cp = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        if (i == 31) {
            ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(1, 0));
        } else {
            ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(0, 0));
        }
    }
}
EXECUTE_TEST(ComputationalBasisState, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateSameSeed(std::uint32_t n) {
    for (std::uint64_t i = 0; i < 3; ++i) {
        auto state1 = StateVector<Fp, Sp>::Haar_random_state(n, i),
             state2 = StateVector<Fp, Sp>::Haar_random_state(n, i);
        ASSERT_TRUE(same_state(state1, state2));
    }
}
EXECUTE_TEST(HaarRandomStateSameSeed, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateWithoutSeed(std::uint32_t n) {
    for (std::uint64_t i = 0; i < 3; ++i) {
        auto state1 = StateVector<Fp, Sp>::Haar_random_state(n, 2 * i),
             state2 = StateVector<Fp, Sp>::Haar_random_state(n, 2 * i + 1);
        ASSERT_FALSE(same_state(state1, state2));
    }
}
EXECUTE_TEST(HaarRandomStateWithoutSeed, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestAddStateWithCoef(std::uint32_t n) {
    const StdComplex<Fp> coef(2.5, 1.3);
    auto state1 = StateVector<Fp, Sp>::Haar_random_state(n);
    auto state2 = StateVector<Fp, Sp>::Haar_random_state(n);
    auto vec1 = state1.get_amplitudes();
    auto vec2 = state2.get_amplitudes();

    state1.add_state_vector_with_coef(coef, state2);
    auto new_vec = state1.get_amplitudes();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
        StdComplex<Fp> res = new_vec[i],
                       val = (StdComplex<Fp>)vec1[i] + coef * (StdComplex<Fp>)vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Fp>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Fp>);
    }
}
EXECUTE_TEST(AddStateWithCoef, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestMultiplyCoef(std::uint32_t n) {
    const StdComplex<Fp> coef(0.5, 0.2);
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    auto vec = state.get_amplitudes();
    state.multiply_coef(coef);
    auto new_vec = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        StdComplex<Fp> res = new_vec[i], val = coef * (StdComplex<Fp>)vec[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Fp>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Fp>);
    }
}
EXECUTE_TEST(MultiplyCoef, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestGetZeroProbability(std::uint32_t n) {
    std::uint32_t dim = 1 << n;
    std::vector<Complex<Fp>> vec(dim);
    std::vector<double> zero_prob(n, 0);
    double denom = (double)(dim - 1) * dim / 2;
    for (std::uint32_t i = 0; i < vec.size(); ++i) {
        vec[i] = std::sqrt(i);
        for (std::uint32_t b = 0; b < n; ++b) {
            if (((i >> b) & 1) == 0) {
                zero_prob[b] += i / denom;
            }
        }
    }
    StateVector<Fp, Sp> state(n);
    state.load(vec);
    state.normalize();
    for (std::uint64_t i = 0; i < n; ++i) {
        ASSERT_NEAR(zero_prob[i], state.get_zero_probability(i), eps<Fp>);
    }
}
EXECUTE_TEST(GetZeroProbability, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestEntropyCalculation(std::uint32_t n) {
    const std::uint64_t dim = 1ULL << n;
    const std::uint64_t max_repeat = 10;

    for (std::uint64_t rep = 0; rep < max_repeat; ++rep) {
        auto state = StateVector<Fp, Sp>::Haar_random_state(n);
        auto state_cp = state.get_amplitudes();
        ASSERT_NEAR(state.get_squared_norm(), 1, eps<Fp>);
        Eigen::Matrix<StdComplex<Fp>, -1, 1> test_state(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state[i] = (StdComplex<Fp>)state_cp[i];

        for (std::uint64_t target = 0; target < n; ++target) {
            Fp ent = 0;
            for (std::uint64_t ind = 0; ind < dim; ++ind) {
                StdComplex<Fp> z = test_state[ind];
                Fp prob = z.real() * z.real() + z.imag() * z.imag();
                ent += -prob * std::log2(prob);
            }
            ASSERT_NEAR(ent, state.get_entropy(), eps<Fp>);
        }
    }
}
EXECUTE_TEST(EntropyCalculation, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestGetMarginalProbability(std::uint32_t n) {
    const std::uint64_t dim = 1 << n;
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    auto state_cp = state.get_amplitudes();
    std::vector<Fp> probs(4, 0);
    for (std::uint64_t i = 0; i < dim; ++i) {
        probs[i & 0b11] += internal::squared_norm(state_cp[i]);
    }

    auto extend = [](std::vector<std::uint64_t> v, std::int64_t n) {
        while ((std::uint32_t)v.size() < n) v.push_back(StateVector<Fp, Sp>::UNMEASURED);
        return v;
    };
    const std::uint64_t U = StateVector<Fp, Sp>::UNMEASURED;
    ASSERT_NEAR(state.get_marginal_probability(extend({0, 0}, n)), probs[0], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({1, 0}, n)), probs[1], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({0, 1}, n)), probs[2], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({1, 1}, n)), probs[3], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({0, U}, n)), probs[0] + probs[2], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({1, U}, n)), probs[1] + probs[3], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({U, 0}, n)), probs[0] + probs[1], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({U, 1}, n)), probs[2] + probs[3], eps<Fp>);
    ASSERT_NEAR(state.get_marginal_probability(extend({U, U}, n)), 1, eps<Fp>);
}
EXECUTE_TEST(GetMarginalProbability, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestSamplingSuperpositionState(std::uint32_t n) {
    const std::uint64_t nshot = 65536;
    const std::uint64_t test_count = 10;
    std::uint64_t pass_count = 0;
    for (std::uint64_t test_i = 0; test_i < test_count; test_i++) {
        StateVector<Fp, Sp> state(n);
        state.set_computational_basis(0);
        for (std::uint64_t i = 1; i <= 4; ++i) {
            StateVector<Fp, Sp> tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_vector_with_coef(1 << i, tmp_state);
        }
        state.normalize();
        std::vector<std::uint64_t> res = state.sampling(nshot);

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
EXECUTE_TEST(SamplingSuperpositionState, 6);

FLOAT_AND_SPACE(Fp, Sp)
void TestSamplingComputationalBasis(std::uint32_t n) {
    const std::uint64_t nshot = 1024;
    StateVector<Fp, Sp> state(n);
    state.set_computational_basis(6);
    auto res = state.sampling(nshot);
    for (std::uint64_t i = 0; i < nshot; ++i) {
        ASSERT_TRUE(res[i] == 6);
    }
}
EXECUTE_TEST(SamplingComputationalBasis, 6);
