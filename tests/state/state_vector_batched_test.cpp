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

TYPED_TEST(StateVectorBatchedTest, SpaceConversionCreatesIndependentCopy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 4, n_qubits = 3;
    auto states =
        StateVectorBatched<Prec, Space>::Haar_random_state(batch_size, n_qubits, false, 0);
    const auto original = states.get_amplitudes();

    auto states_default = states.copy_to_default_space();
    auto states_host = states.copy_to_host_space();
    auto default_amp = states_default.get_amplitudes();
    auto host_amp = states_host.get_amplitudes();

    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < states.dim(); ++i) {
            ASSERT_NEAR(default_amp[b][i].real(), original[b][i].real(), eps<Prec>);
            ASSERT_NEAR(default_amp[b][i].imag(), original[b][i].imag(), eps<Prec>);
            ASSERT_NEAR(host_amp[b][i].real(), original[b][i].real(), eps<Prec>);
            ASSERT_NEAR(host_amp[b][i].imag(), original[b][i].imag(), eps<Prec>);
        }
    }

    states_default.view_state_vector_at(0).set_amplitude_at(0, StdComplex(0.25, -0.5));
    auto unchanged = states.get_state_vector_at(0).get_amplitude_at(0);
    ASSERT_NEAR(unchanged.real(), original[0][0].real(), eps<Prec>);
    ASSERT_NEAR(unchanged.imag(), original[0][0].imag(), eps<Prec>);
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

TYPED_TEST(StateVectorBatchedTest, ViewStateVectorAtSharesStorage) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 2, n_qubits = 3;
    StateVectorBatched<Prec, Space> states(batch_size, n_qubits);
    states.set_zero_norm_state();

    auto view = states.view_state_vector_at(1);
    const StdComplex changed(0.25, -0.5);
    view.set_amplitude_at(3, changed);

    auto copied_from_batched = states.get_state_vector_at(1);
    auto got = copied_from_batched.get_amplitude_at(3);
    ASSERT_NEAR(got.real(), changed.real(), eps<Prec>);
    ASSERT_NEAR(got.imag(), changed.imag(), eps<Prec>);

    auto another_batch = states.get_state_vector_at(0);
    auto another_amp = another_batch.get_amplitude_at(3);
    ASSERT_NEAR(another_amp.real(), 0., eps<Prec>);
    ASSERT_NEAR(another_amp.imag(), 0., eps<Prec>);
}

TYPED_TEST(StateVectorBatchedTest, CopyStateVectorAtDoesNotAliasStorage) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 2, n_qubits = 3;
    StateVectorBatched<Prec, Space> states(batch_size, n_qubits);
    states.set_zero_norm_state();

    auto copied = states.get_state_vector_at(1);
    const StdComplex changed(0.75, 0.125);
    copied.set_amplitude_at(2, changed);

    auto original = states.get_state_vector_at(1);
    auto got = original.get_amplitude_at(2);
    ASSERT_NEAR(got.real(), 0., eps<Prec>);
    ASSERT_NEAR(got.imag(), 0., eps<Prec>);
}

TYPED_TEST(StateVectorBatchedTest, BatchIdBoundsCheck) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    const std::uint64_t batch_size = 2, n_qubits = 3;
    StateVectorBatched<Prec, Space> states(batch_size, n_qubits);
    StateVector<Prec, Space> state(n_qubits);

    EXPECT_NO_THROW((void)states.view_state_vector_at(batch_size - 1));
    EXPECT_NO_THROW((void)states.get_state_vector_at(batch_size - 1));
    EXPECT_NO_THROW(states.set_state_vector_at(batch_size - 1, state));

    EXPECT_THROW((void)states.view_state_vector_at(batch_size), std::runtime_error);
    EXPECT_THROW((void)states.get_state_vector_at(batch_size), std::runtime_error);
    EXPECT_THROW(states.set_state_vector_at(batch_size, state), std::runtime_error);
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

    auto entropies = states.get_computational_basis_entropy();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto state = states.get_state_vector_at(b);
        ASSERT_NEAR(entropies[b], state.get_computational_basis_entropy(), eps<Prec>);
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

TYPED_TEST(StateVectorBatchedTest, WeightedBatchAlgebra) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    using Batch = StateVectorBatched<Prec, Space>;
    const std::vector<StdComplex> coefs{{0.5, -0.25}, {-0.25, 0.5}, {0., 0.}};
    const std::vector<std::vector<StdComplex>> input{
        {{0.25, 0.5}, {-0.5, 0.25}}, {{-0.25, 0.25}, {0.5, -0.5}}, {{0.5, 0.}, {0., 0.5}}};
    Batch states(3, 1);
    states.load(input);
    const auto combined = states.linear_combination(coefs).get_amplitudes();
    for (std::size_t i = 0; i < 2; ++i) {
        StdComplex expected = 0.;
        for (std::size_t b = 0; b < 3; ++b) expected += coefs[b] * input[b][i];
        ASSERT_NEAR(std::abs(combined[i] - expected), 0., eps<Prec>);
    }
    auto scaled = states.copy();
    scaled.multiply_coef(coefs);
    auto products = Batch::inner_product(states, scaled);
    auto scaled_amp = scaled.get_amplitudes();
    for (std::size_t b = 0; b < 3; ++b) {
        StdComplex expected_product = 0.;
        for (std::size_t i = 0; i < 2; ++i) {
            ASSERT_NEAR(std::abs(scaled_amp[b][i] - coefs[b] * input[b][i]), 0., eps<Prec>);
            expected_product += std::conj(input[b][i]) * coefs[b] * input[b][i];
        }
        ASSERT_NEAR(std::abs(products[b] - expected_product), 0., eps<Prec>);
    }
    states.add_state_vector_with_coef(coefs, states);
    auto self_added = states.get_amplitudes();
    for (std::size_t b = 0; b < 3; ++b) {
        for (std::size_t i = 0; i < 2; ++i) {
            ASSERT_NEAR(std::abs(self_added[b][i] - (1. + coefs[b]) * input[b][i]), 0., eps<Prec>);
        }
    }
    EXPECT_THROW(states.multiply_coef(std::vector<StdComplex>{}), std::invalid_argument);
    EXPECT_THROW((void)states.linear_combination(std::vector<StdComplex>{}), std::invalid_argument);
    Batch wrong_batch(2, 1), wrong_dim(3, 2);
    EXPECT_THROW(states.add_state_vector_with_coef(coefs, wrong_batch), std::invalid_argument);
    EXPECT_THROW((void)Batch::inner_product(states, wrong_dim), std::invalid_argument);
    Batch empty(0, 1);
    auto zero = empty.linear_combination({}).get_amplitudes();
    EXPECT_EQ(zero, std::vector<StdComplex>(2, 0.));
    EXPECT_TRUE(Batch::inner_product(empty, empty).empty());
}

TYPED_TEST(StateVectorBatchedTest, WeightedPartialOverlapAndZeroQubits) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    using Batch = StateVectorBatched<Prec, Space>;
    Batch storage(3, 0);
    storage.load({{{1., 0.}}, {{2., 0.}}, {{3., 0.}}});
    Batch src(2, 0), dst(2, 0);
    src._raw = Kokkos::subview(storage._raw, std::make_pair(0, 2), Kokkos::ALL());
    dst._raw = Kokkos::subview(storage._raw, std::make_pair(1, 3), Kokkos::ALL());
    dst.add_state_vector_with_coef(std::vector<StdComplex>{1., 1.}, src);
    auto actual = storage.get_amplitudes();
    EXPECT_EQ(actual[0][0], StdComplex(1, 0));
    EXPECT_EQ(actual[1][0], StdComplex(3, 0));
    EXPECT_EQ(actual[2][0], StdComplex(5, 0));
    auto before = dst.get_amplitudes();
    EXPECT_THROW(dst.multiply_coef(std::vector<StdComplex>{1.}), std::invalid_argument);
    EXPECT_EQ(dst.get_amplitudes(), before);
    EXPECT_THROW(dst.add_state_vector_with_coef(std::vector<StdComplex>{1.}, src),
                 std::invalid_argument);
    EXPECT_EQ(dst.get_amplitudes(), before);
    Batch empty(0, 0);
    EXPECT_EQ(empty.linear_combination({}).get_amplitudes(), (std::vector<StdComplex>{0.}));
    EXPECT_TRUE(Batch::inner_product(empty, empty).empty());
}

TYPED_TEST(StateVectorBatchedTest, WeightedSumAccumulatesAtLeastInFloat) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    if constexpr (Prec == Precision::F16 || Prec == Precision::BF16) {
        // The final value is representable in both formats; low-precision accumulation
        // would stop changing well before all 4096 terms have been added.
        StateVectorBatched<Prec, Space> states(4096, 0);
        auto sum = states.linear_combination(std::vector<StdComplex>(4096, 1.));
        ASSERT_EQ(sum.get_amplitude_at(0), StdComplex(4096., 0.));
    }
}
