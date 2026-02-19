#include <gtest/gtest.h>

#include <scaluq/operator/operator_batched.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class OperatorBatchedTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(OperatorBatchedTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
std::pair<OperatorBatched<Prec, Space>, std::vector<Operator<Prec, Space>>>
generate_random_observable(int n) {
    Random random;
    std::uint64_t batch_size = random.int32() % 5 + 1;
    std::vector<std::vector<PauliOperator<Prec>>> rand_observable;
    std::vector<Operator<Prec, Space>> test_rand_observable;

    for (std::uint64_t b = 0; b < batch_size; ++b) {
        std::vector<PauliOperator<Prec>> ops;
        std::uint64_t term_count = random.int32() % 10 + 1;
        for (std::uint64_t term = 0; term < term_count; ++term) {
            std::vector<std::uint64_t> paulis(n, 0);
            double coef = random.uniform();
            for (std::uint64_t i = 0; i < paulis.size(); ++i) {
                paulis[i] = random.int32() % 4;
            }

            std::string str = "";
            for (std::uint64_t ind = 0; ind < paulis.size(); ind++) {
                std::uint64_t val = paulis[ind];
                if (val != 0) {
                    if (val == 1)
                        str += " X";
                    else if (val == 2)
                        str += " Y";
                    else if (val == 3)
                        str += " Z";
                    str += " " + std::to_string(ind);
                }
            }
            ops.push_back(PauliOperator<Prec>(str.c_str(), coef));
        }
        rand_observable.push_back(ops);
        test_rand_observable.push_back(Operator<Prec, Space>(ops));
    }
    return {OperatorBatched<Prec, Space>(rand_observable), std::move(test_rand_observable)};
}

TYPED_TEST(OperatorBatchedTest, TransitionAmplitudes) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;
    auto [op_batched, ops] = generate_random_observable<Prec, Space>(n);
    StateVector<Prec, Space> state_vector_bra(n);
    state_vector_bra.set_Haar_random_state();
    StateVector<Prec, Space> state_vector_ket(n);
    state_vector_ket.set_Haar_random_state();

    auto res_batched = op_batched.get_transition_amplitude(state_vector_bra, state_vector_ket);
    for (std::uint64_t b = 0; b < op_batched.batch_size(); ++b) {
        auto res = ops[b].get_transition_amplitude(state_vector_bra, state_vector_ket);
        ASSERT_NEAR(res.real(), res_batched[b].real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), res_batched[b].imag(), eps<Prec>);
    }
}

TYPED_TEST(OperatorBatchedTest, ExpectationValues) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;
    auto [op_batched, ops] = generate_random_observable<Prec, Space>(n);
    StateVector<Prec, Space> state_vector(n);
    state_vector.set_Haar_random_state();

    auto res_batched = op_batched.get_expectation_value(state_vector);
    for (std::uint64_t b = 0; b < op_batched.batch_size(); ++b) {
        auto res = ops[b].get_expectation_value(state_vector);
        ASSERT_NEAR(res.real(), res_batched[b].real(), eps<Prec>);
        ASSERT_NEAR(res.imag(), res_batched[b].imag(), eps<Prec>);
    }
}

TYPED_TEST(OperatorBatchedTest, Copy) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;
    auto [op_batched, ops] = generate_random_observable<Prec, Space>(n);
    OperatorBatched<Prec, Space> op_batched_copy = op_batched.copy();
    EXPECT_EQ(op_batched.to_string(), op_batched_copy.to_string());
}

TYPED_TEST(OperatorBatchedTest, ViewOperatorAtSharesStorage) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    OperatorBatched<Prec, Space> op_batched(
        std::vector<std::vector<PauliOperator<Prec>>>{{PauliOperator<Prec>("X 0", 1.)},
                                                      {PauliOperator<Prec>("Z 0", 2.)}});

    auto view = op_batched.view_operator_at(op_batched.concurrent_stream(), 0);
    view *= StdComplex(3., 0.);

    auto copied = op_batched.get_operator_at(0);
    auto terms = copied.get_terms();
    ASSERT_EQ(terms.size(), 1);
    ASSERT_NEAR(terms[0].coef().real(), 3., eps<Prec>);
    ASSERT_NEAR(terms[0].coef().imag(), 0., eps<Prec>);
}

TYPED_TEST(OperatorBatchedTest, CopyOperatorAtDoesNotAliasStorage) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    OperatorBatched<Prec, Space> op_batched(
        std::vector<std::vector<PauliOperator<Prec>>>{{PauliOperator<Prec>("X 0", 1.)},
                                                      {PauliOperator<Prec>("Z 0", 2.)}});

    auto copied = op_batched.get_operator_at(op_batched.concurrent_stream(), 1);
    copied *= StdComplex(5., 0.);

    auto original = op_batched.get_operator_at(op_batched.concurrent_stream(), 1);
    auto terms = original.get_terms();
    ASSERT_EQ(terms.size(), 1);
    ASSERT_NEAR(terms[0].coef().real(), 2., eps<Prec>);
    ASSERT_NEAR(terms[0].coef().imag(), 0., eps<Prec>);
}

TYPED_TEST(OperatorBatchedTest, ViewOperatorAtForbidsOptimize) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    OperatorBatched<Prec, Space> op_batched(
        std::vector<std::vector<PauliOperator<Prec>>>{{PauliOperator<Prec>("X 0", 1.)}});
    auto view = op_batched.view_operator_at(0);
    EXPECT_THROW(view.optimize(), std::runtime_error);
}

TYPED_TEST(OperatorBatchedTest, Apply) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;
    auto [op_batched, ops] = generate_random_observable<Prec, Space>(n);
    StateVector<Prec, Space> state_vector(n);
    state_vector.set_Haar_random_state();
    auto states_batched = op_batched.get_applied_states(state_vector);
    for (std::uint64_t b = 0; b < op_batched.batch_size(); ++b) {
        StateVector<Prec, Space> state_vector_single = state_vector.copy();
        ops[b].apply_to_state(state_vector_single);
        auto amp_batched = states_batched.get_state_vector_at(b).get_amplitudes();
        auto amp_single = state_vector_single.get_amplitudes();
        ASSERT_EQ(amp_batched.size(), amp_single.size());
        for (std::uint64_t i = 0; i < amp_batched.size(); ++i) {
            ASSERT_NEAR(amp_batched[i].real(), amp_single[i].real(), eps<Prec>);
            ASSERT_NEAR(amp_batched[i].imag(), amp_single[i].imag(), eps<Prec>);
        }
    }
}

TYPED_TEST(OperatorBatchedTest, Json) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    std::uint64_t n = 4;
    Random random;
    auto [op_batched, ops] = generate_random_observable<Prec, Space>(n);
    Json j = op_batched;
    OperatorBatched<Prec, Space> op_batched_from_json = j;
    EXPECT_EQ(op_batched.to_string(), op_batched_from_json.to_string());
}
