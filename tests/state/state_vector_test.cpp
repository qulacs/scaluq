#include <gtest/gtest.h>

#include <scaluq/state/state_vector.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

<<<<<<< HEAD
template <typename T>
class StateVectorTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(StateVectorTest, TestTypes, NameGenerator);

TYPED_TEST(StateVectorTest, HaarRandomStateNorm) {
    constexpr Precision Prec = TestFixture::Prec;
    const int n_tries = 10;
    for (int n = 1; n <= n_tries; n++) {
        const auto state = StateVector<Prec>::Haar_random_state(n);
        ASSERT_NEAR(state.get_squared_norm(), 1., eps<Prec>);
    }
}

TYPED_TEST(StateVectorTest, OperationAtIndex) {
    constexpr Precision Prec = TestFixture::Prec;
    auto state = StateVector<Prec>::Haar_random_state(10);
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        state.set_amplitude_at(i, 1);
        ASSERT_NEAR(state.get_amplitude_at(i).real(), 1., eps<Prec>);
        ASSERT_NEAR(state.get_amplitude_at(i).imag(), 0., eps<Prec>);
=======
#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>
#define EXECUTE_TEST(Name, arg)                \
    TEST(StateVectorTest, Name) {              \
        Test##Name<double, DefaultSpace>(arg); \
        Test##Name<double, HostSpace>(arg);    \
        Test##Name<float, DefaultSpace>(arg);  \
        Test##Name<float, HostSpace>(arg);     \
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
>>>>>>> set-space
    }
}
EXECUTE_TEST(OperationAtIndex, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, CopyState) {
    constexpr Precision Prec = TestFixture::Prec;
    const int n = 5;
    const auto state = StateVector<Prec>::Haar_random_state(n);
    StateVector state_cp = state.copy();
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestCopyState(std::uint32_t n) {
    const auto state = StateVector<Fp, Sp>::Haar_random_state(n);
    StateVector<Fp, Sp> state_cp = state.copy();
>>>>>>> set-space
    auto vec1 = state.get_amplitudes();
    auto vec2 = state_cp.get_amplitudes();
    ASSERT_EQ(vec1, vec2);
}
EXECUTE_TEST(CopyState, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, ZeroNormState) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 5;

    StateVector state(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestZeroNormState(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
>>>>>>> set-space
    state.set_zero_norm_state();
    auto state_cp = state.get_amplitudes();
    for (std::uint64_t i = 0; i < state.dim(); ++i) {
<<<<<<< HEAD
        ASSERT_EQ((StdComplex)state_cp[i], StdComplex(0, 0));
=======
        ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(0, 0));
>>>>>>> set-space
    }
}
EXECUTE_TEST(ZeroNormState, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, ComputationalBasisState) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 5;

    StateVector state(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestComputationalBasisState(std::uint32_t n) {
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
>>>>>>> set-space
    state.set_computational_basis(31);
    auto state_cp = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
        if (i == 31) {
<<<<<<< HEAD
            ASSERT_EQ((StdComplex)state_cp[i], StdComplex(1, 0));
        } else {
            ASSERT_EQ((StdComplex)state_cp[i], StdComplex(0, 0));
=======
            ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(1, 0));
        } else {
            ASSERT_EQ((StdComplex<Fp>)state_cp[i], StdComplex<Fp>(0, 0));
>>>>>>> set-space
        }
    }
}
EXECUTE_TEST(ComputationalBasisState, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, HaarRandomStateSameSeed) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10, m = 5;
    for (std::uint64_t i = 0; i < m; ++i) {
        StateVector state1(StateVector<Prec>::Haar_random_state(n, i)),
            state2(StateVector<Prec>::Haar_random_state(n, i));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateSameSeed(std::uint32_t n) {
    for (std::uint64_t i = 0; i < 3; ++i) {
        auto state1 = StateVector<Fp, Sp>::Haar_random_state(n, i),
             state2 = StateVector<Fp, Sp>::Haar_random_state(n, i);
>>>>>>> set-space
        ASSERT_TRUE(same_state(state1, state2));
    }
}
EXECUTE_TEST(HaarRandomStateSameSeed, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, HaarRandomStateWithoutSeed) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10, m = 5;
    for (std::uint64_t i = 0; i < m; ++i) {
        StateVector state1(StateVector<Prec>::Haar_random_state(n)),
            state2(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateWithoutSeed(std::uint32_t n) {
    for (std::uint64_t i = 0; i < 3; ++i) {
        auto state1 = StateVector<Fp, Sp>::Haar_random_state(n, 2 * i),
             state2 = StateVector<Fp, Sp>::Haar_random_state(n, 2 * i + 1);
>>>>>>> set-space
        ASSERT_FALSE(same_state(state1, state2));
    }
}
EXECUTE_TEST(HaarRandomStateWithoutSeed, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, AddState) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10;
    StateVector state1(StateVector<Prec>::Haar_random_state(n));
    StateVector state2(StateVector<Prec>::Haar_random_state(n));
    auto vec1 = state1.get_amplitudes();
    auto vec2 = state2.get_amplitudes();
    state1.add_state_vector_with_coef(1, state2);
    auto new_vec = state1.get_amplitudes();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
        StdComplex res = new_vec[i], val = vec1[i] + vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Prec>);
    }
}

TYPED_TEST(StateVectorTest, AddStateWithCoef) {
    constexpr Precision Prec = TestFixture::Prec;
    const StdComplex coef(2.5, 1.3);
    const std::uint64_t n = 10;
    StateVector state1(StateVector<Prec>::Haar_random_state(n));
    StateVector state2(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestAddStateWithCoef(std::uint32_t n) {
    const StdComplex<Fp> coef(2.5, 1.3);
    auto state1 = StateVector<Fp, Sp>::Haar_random_state(n);
    auto state2 = StateVector<Fp, Sp>::Haar_random_state(n);
>>>>>>> set-space
    auto vec1 = state1.get_amplitudes();
    auto vec2 = state2.get_amplitudes();

    state1.add_state_vector_with_coef(coef, state2);
    auto new_vec = state1.get_amplitudes();

    for (std::uint64_t i = 0; i < state1.dim(); ++i) {
<<<<<<< HEAD
        StdComplex res = new_vec[i], val = vec1[i] + coef * vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Prec>);
=======
        StdComplex<Fp> res = new_vec[i],
                       val = (StdComplex<Fp>)vec1[i] + coef * (StdComplex<Fp>)vec2[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Fp>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Fp>);
>>>>>>> set-space
    }
}
EXECUTE_TEST(AddStateWithCoef, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, MultiplyCoef) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10;
    const StdComplex coef(0.5, 0.2);

    StateVector state(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestMultiplyCoef(std::uint32_t n) {
    const StdComplex<Fp> coef(0.5, 0.2);
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
>>>>>>> set-space
    auto vec = state.get_amplitudes();
    state.multiply_coef(coef);
    auto new_vec = state.get_amplitudes();

    for (std::uint64_t i = 0; i < state.dim(); ++i) {
<<<<<<< HEAD
        StdComplex res = new_vec[i], val = coef * vec[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Prec>);
=======
        StdComplex<Fp> res = new_vec[i], val = coef * (StdComplex<Fp>)vec[i];
        ASSERT_NEAR(res.real(), val.real(), eps<Fp>);
        ASSERT_NEAR(res.imag(), val.imag(), eps<Fp>);
>>>>>>> set-space
    }
}
EXECUTE_TEST(MultiplyCoef, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, GetZeroProbability) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10;
    StateVector<Prec> state(n);
    state.set_computational_basis(1);
    for (std::uint64_t i = 2; i <= 10; ++i) {
        StateVector<Prec> tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_vector_with_coef(std::sqrt(i), tmp_state);
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestGetZeroProbability(std::uint32_t n) {
    std::uint32_t dim = 1 << n;
    std::vector<Complex<Fp> > vec(dim);
    std::vector<double> zero_prob(n, 0);
    double denom = (double)(dim - 1) * dim / 2;
    for (std::uint32_t i = 0; i < vec.size(); ++i) {
        vec[i] = std::sqrt(i);
        for (std::uint32_t b = 0; b < n; ++b) {
            if (((i >> b) & 1) == 0) {
                zero_prob[b] += i / denom;
            }
        }
>>>>>>> set-space
    }
    StateVector<Fp, Sp> state(n);
    state.load(vec);
    state.normalize();
<<<<<<< HEAD
    ASSERT_NEAR(state.get_zero_probability(0), 30.0 / 55.0, eps<Prec>);
    ASSERT_NEAR(state.get_zero_probability(1), 27.0 / 55.0, eps<Prec>);
    ASSERT_NEAR(state.get_zero_probability(2), 33.0 / 55.0, eps<Prec>);
    ASSERT_NEAR(state.get_zero_probability(3), 28.0 / 55.0, eps<Prec>);
=======
    for (std::uint64_t i = 0; i < n; ++i) {
        ASSERT_NEAR(zero_prob[i], state.get_zero_probability(i), eps<Fp>);
    }
>>>>>>> set-space
}
EXECUTE_TEST(GetZeroProbability, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, EntropyCalculation) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 6;
    const std::uint64_t dim = 1ULL << n;
    const std::uint64_t max_repeat = 10;

    StateVector<Prec> state(n);
    for (std::uint64_t rep = 0; rep < max_repeat; ++rep) {
        state = StateVector<Prec>::Haar_random_state(n);
        auto state_cp = state.get_amplitudes();
        ASSERT_NEAR(state.get_squared_norm(), 1, eps<Prec>);
        Eigen::VectorXcd test_state(dim);
        for (std::uint64_t i = 0; i < dim; ++i) test_state[i] = state_cp[i];
=======
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
>>>>>>> set-space

        for (std::uint64_t target = 0; target < n; ++target) {
            Fp ent = 0;
            for (std::uint64_t ind = 0; ind < dim; ++ind) {
<<<<<<< HEAD
                StdComplex z = test_state[ind];
                double prob = z.real() * z.real() + z.imag() * z.imag();
                if (prob > 0.) ent += -prob * std::log2(prob);
            }
            ASSERT_NEAR(ent, state.get_entropy(), eps<Prec>);
=======
                StdComplex<Fp> z = test_state[ind];
                Fp prob = z.real() * z.real() + z.imag() * z.imag();
                ent += -prob * std::log2(prob);
            }
            ASSERT_NEAR(ent, state.get_entropy(), eps<Fp>);
>>>>>>> set-space
        }
    }
}
EXECUTE_TEST(EntropyCalculation, 6);

<<<<<<< HEAD
TYPED_TEST(StateVectorTest, GetMarginalProbability) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 2;
    const std::uint64_t dim = 1 << n;
    StateVector state(StateVector<Prec>::Haar_random_state(n));
=======
FLOAT_AND_SPACE(Fp, Sp)
void TestGetMarginalProbability(std::uint32_t n) {
    const std::uint64_t dim = 1 << n;
    auto state = StateVector<Fp, Sp>::Haar_random_state(n);
>>>>>>> set-space
    auto state_cp = state.get_amplitudes();
    std::vector<Fp> probs(4, 0);
    for (std::uint64_t i = 0; i < dim; ++i) {
<<<<<<< HEAD
        probs.push_back(std::norm(state_cp[i]));
    }
    ASSERT_NEAR(state.get_marginal_probability({0, 0}), probs[0], eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({1, 0}), probs[1], eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({0, 1}), probs[2], eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({1, 1}), probs[3], eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({0, StateVector<Prec>::UNMEASURED}),
                probs[0] + probs[2],
                eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({1, StateVector<Prec>::UNMEASURED}),
                probs[1] + probs[3],
                eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({StateVector<Prec>::UNMEASURED, 0}),
                probs[0] + probs[1],
                eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability({StateVector<Prec>::UNMEASURED, 1}),
                probs[2] + probs[3],
                eps<Prec>);
    ASSERT_NEAR(state.get_marginal_probability(
                    {StateVector<Prec>::UNMEASURED, StateVector<Prec>::UNMEASURED}),
                1.,
                eps<Prec>);
}

TYPED_TEST(StateVectorTest, SamplingSuperpositionState) {
    constexpr Precision Prec = TestFixture::Prec;
    const std::uint64_t n = 10;
    const std::uint64_t nshot = 65536;
    const std::uint64_t test_count = 10;
    std::uint64_t pass_count = 0;
    for (std::uint64_t test_i = 0; test_i < test_count; test_i++) {
        StateVector<Prec> state(n);
        state.set_computational_basis(0);
        for (std::uint64_t i = 1; i <= 4; ++i) {
            StateVector<Prec> tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_vector_with_coef(1 << i, tmp_state);
        }
        state.normalize();
        std::vector<std::uint64_t> res = state.sampling(nshot);

        std::array<std::uint64_t, 5> cnt = {};
        for (std::uint64_t i = 0; i < nshot; ++i) {
            ASSERT_GE(res[i], 0);
            ASSERT_LE(res[i], (1 << n) - 1);
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
=======
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
    StateVector<Fp, Sp> state(n);
    state.set_computational_basis(0);
    for (std::uint64_t i = 1; i <= 4; ++i) {
        StateVector<Fp, Sp> tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_vector_with_coef(1 << i, tmp_state);
>>>>>>> set-space
    }
    state.normalize();
    std::vector<std::uint64_t> res = state.sampling(nshot);

<<<<<<< HEAD
    TYPED_TEST(StateVectorTest, SamplingComputationalBasis) {
        constexpr Precision Prec = TestFixture::Prec;
        const std::uint64_t n = 10;
        const std::uint64_t nshot = 1024;
        StateVector<Prec> state(n);
        state.set_computational_basis(100);
        auto res = state.sampling(nshot);
        for (std::uint64_t i = 0; i < nshot; ++i) {
            ASSERT_EQ(res[i], 100);
=======
    std::array<std::uint64_t, 5> cnt = {};
    for (std::uint64_t i = 0; i < nshot; ++i) {
        ASSERT_GE(res[i], 0);
        ASSERT_LE(res[i], 4);
        cnt[res[i]] += 1;
    }
    bool pass = true;
    for (std::uint64_t i = 0; i < 4; i++) {
        ASSERT_GT(cnt[i + 1], cnt[i]);
    }
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
>>>>>>> set-space
        }
    }
    EXECUTE_TEST(SamplingComputationalBasis, 6);
