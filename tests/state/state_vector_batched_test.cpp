#include <gtest/gtest.h>

#include <scaluq/state/state_vector_batched.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class StateVectorBatchedTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(StateVectorBatchedTest, TestTypes, NameGenerator);

TYPED_TEST(StateVectorBatchedTest, HaarRandomStateNorm) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 10, n_qubits = 3;
    const auto states =
        StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);
    auto norms = states.get_squared_norm();
    for (auto x : norms) ASSERT_NEAR(x, 1., eps<Prec>);
}

TYPED_TEST(StateVectorBatchedTest, LoadAndAmplitudes) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 3;
    const std::uint64_t dim = 1 << n_qubits;
    std::vector states_h(batch_size, std::vector<StdComplex>(dim));
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < dim; ++i) {
            states_h[b][i] = b * dim + i;
        }
    }
    StateVectorBatched<Prec, Space> states(batch_size, n_qubits);

    states.load(states_h);
    auto amps = states.get_amplitudes();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < dim; ++i) {
            ASSERT_EQ(amps[b][i].real(), b * dim + i);
        }
    }
}

TYPED_TEST(StateVectorBatchedTest, OperateState) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);
    auto states_add =
        StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);
    const StdComplex coef(2.1, 3.5);

    auto states_cp = states.copy();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        ASSERT_TRUE(same_state(states.get_state_vector_at(b), states_cp.get_state_vector_at(b)));
    }

    states.add_state_vector_with_coef(1, states_add);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at(b);
        v.add_state_vector_with_coef(1, states_add.get_state_vector_at(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector_at(b)));
    }

    states_cp = states.copy();
    states.add_state_vector_with_coef(coef, states_add);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at(b);
        v.add_state_vector_with_coef(coef, states_add.get_state_vector_at(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector_at(b)));
    }

    states_cp = states.copy();
    states.multiply_coef(coef);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at(b);
        v.multiply_coef(coef);
        ASSERT_TRUE(same_state(v, states.get_state_vector_at(b)));
    }
}

TYPED_TEST(StateVectorBatchedTest, ZeroProbs) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);

    for (std::uint64_t i = 0; i < n_qubits; ++i) {
        auto zero_probs = states.get_zero_probability(i);
        for (std::uint64_t b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector_at(b);
            ASSERT_NEAR(zero_probs[b], state.get_zero_probability(i), eps<Prec>);
        }
    }
}

TYPED_TEST(StateVectorBatchedTest, MarginalProbs) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 5;
    auto states = StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);

    Random rd(0);
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::vector<std::uint64_t> targets;
        for (std::uint64_t j = 0; j < n_qubits; ++j) {
            targets.push_back(rd.int32() % 3);
        }
        auto mg_probs = states.get_marginal_probability(targets);
        for (std::uint64_t b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector_at(b);
            ASSERT_NEAR(mg_probs[b], state.get_marginal_probability(targets), eps<Prec>);
        }
    }
}

TYPED_TEST(StateVectorBatchedTest, Entropy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 3;
    auto states = StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false);

    auto entropies = states.get_entropy();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto state = states.get_state_vector_at(b);
        ASSERT_NEAR(entropies[b], state.get_entropy(), eps<Prec>);
    }
}

TYPED_TEST(StateVectorBatchedTest, Sampling) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 2, n_qubits = 3;
    StateVectorBatched<Prec, Space> states(batch_size, n_qubits);
    states.load(
        std::vector<std::vector<StdComplex>>{{1, 4, 5, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 6, 4, 1}});
    states.normalize();
    auto result = states.sampling(4096);
    std::vector cnt(2, std::vector<std::uint64_t>(states.dim(), 0));
    for (std::uint64_t i = 0; i < 2; ++i) {
        for (auto x : result[i]) {
            ++cnt[i][x];
        }
    }
    ASSERT_LT(cnt[0][0], cnt[0][1]);
    ASSERT_LT(cnt[0][1], cnt[0][2]);
    ASSERT_EQ(cnt[0][3], 0);
    ASSERT_EQ(cnt[0][4], 0);
    ASSERT_EQ(cnt[0][5], 0);
    ASSERT_EQ(cnt[0][6], 0);
    ASSERT_EQ(cnt[0][7], 0);
    ASSERT_GT(cnt[1][5], cnt[1][6]);
    ASSERT_GT(cnt[1][6], cnt[1][7]);
    ASSERT_EQ(cnt[1][0], 0);
    ASSERT_EQ(cnt[1][1], 0);
    ASSERT_EQ(cnt[1][2], 0);
    ASSERT_EQ(cnt[1][3], 0);
    ASSERT_EQ(cnt[1][4], 0);
}
