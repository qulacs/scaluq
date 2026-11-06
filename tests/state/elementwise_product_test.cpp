#include <gtest/gtest.h>

#include <scaluq/state/state_vector.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class ElementwiseProductTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(ElementwiseProductTest, TestTypes, NameGenerator);

TYPED_TEST(ElementwiseProductTest, ComplexDiagonalAndSelfAlias) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    using State = StateVector<Prec, Space>;
    const std::vector<StdComplex> input{{.25, -.5}, {-.5, .25}, {0., 0.}, {1., .5}};
    const std::vector<StdComplex> weights{{2., 0.}, {0., -1.}, {0., 0.}, {-.5, .5}};
    State state(2), diagonal(2);
    state.load(input);
    diagonal.load(weights);
    state.multiply_elementwise(diagonal);
    const auto result = state.get_amplitudes();
    const auto unchanged = diagonal.get_amplitudes();
    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(std::abs(result[i] - input[i] * weights[i]), 0., eps<Prec>);
        EXPECT_NEAR(std::abs(unchanged[i] - weights[i]), 0., eps<Prec>);
    }
    state.load(input);
    auto alias = state;  // A distinct object sharing the same amplitudes.
    state.multiply_elementwise(alias);
    const auto squared = state.get_amplitudes();
    for (std::size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(std::abs(squared[i] - input[i] * input[i]), 0., eps<Prec>);
    }
}

TYPED_TEST(ElementwiseProductTest, ScalarAndInvalidDimensions) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    using State = StateVector<Prec, Space>;
    State scalar(0), diagonal(0), wrong(1);
    scalar.load(std::vector<StdComplex>{{.5, -.5}});
    diagonal.load(std::vector<StdComplex>{{2., 0.}});
    scalar.multiply_elementwise(diagonal);
    EXPECT_NEAR(std::abs(scalar.get_amplitudes()[0] - StdComplex(1., -1.)), 0., eps<Prec>);
    EXPECT_THROW(scalar.multiply_elementwise(wrong), std::invalid_argument);
    EXPECT_NEAR(std::abs(scalar.get_amplitudes()[0] - StdComplex(1., -1.)), 0., eps<Prec>);
}

TYPED_TEST(ElementwiseProductTest, PartiallyOverlappingViews) {
    constexpr auto Prec = TestFixture::Prec;
    constexpr auto Space = TestFixture::Space;
    using State = StateVector<Prec, Space>;
    typename State::RawView storage("overlapping_states", 48);
    auto host = Kokkos::create_mirror_view(storage);
    for (std::uint64_t i = 0; i < 48; ++i) {
        host(i) = internal::Complex<Prec>(.125 * (1 + i % 4), .25);
    }
    for (bool reversed : {false, true}) {
        Kokkos::deep_copy(storage, host);
        State left(Kokkos::subview(storage, std::make_pair(0ULL, 32ULL)));
        State right(Kokkos::subview(storage, std::make_pair(16ULL, 48ULL)));
        auto& output = reversed ? right : left;
        auto& diagonal = reversed ? left : right;
        const auto original = output.get_amplitudes();
        const auto weights = diagonal.get_amplitudes();
        output.multiply_elementwise(diagonal);
        const auto result = output.get_amplitudes();
        for (std::size_t i = 0; i < result.size(); ++i) {
            EXPECT_NEAR(std::abs(result[i] - original[i] * weights[i]), 0., eps<Prec>);
        }
    }
}
