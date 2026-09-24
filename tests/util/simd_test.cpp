#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <scaluq/util/simd.hpp>
#include <type_traits>

namespace {
namespace simd_ops = scaluq::internal::simd_ops;
namespace sx = Kokkos::Experimental;

template <typename Scalar>
using Bits = std::conditional_t<sizeof(Scalar) == 4, std::uint32_t, std::uint64_t>;

template <typename Scalar>
auto special_values() {
    const auto nan = std::bit_cast<Scalar>(
        std::bit_cast<Bits<Scalar>>(std::numeric_limits<Scalar>::quiet_NaN()) |
        Bits<Scalar>{0x123});
    return std::array<Scalar, 8>{Scalar{0},
                                 -Scalar{0},
                                 Scalar{1.25},
                                 Scalar{-2.5},
                                 std::numeric_limits<Scalar>::infinity(),
                                 -std::numeric_limits<Scalar>::infinity(),
                                 nan,
                                 -nan};
}

template <typename Scalar, std::size_t... Indices>
void check_permutation(std::index_sequence<Indices...> permutation) {
    using Simd = simd_ops::Simd<Scalar>;
    constexpr std::size_t lanes = Simd::size();
    constexpr std::array<std::size_t, lanes> indices{Indices...};
    const auto values = special_values<Scalar>();
    // Rotate special values for scalar ABIs, then use distinct values across all lanes.
    for (std::size_t offset = 0; offset <= values.size(); ++offset) {
        std::array<Scalar, lanes> input, output;
        for (std::size_t i = 0; i < lanes; ++i) {
            input[i] = offset < values.size() ? values[(i + offset) % values.size()]
                                              : static_cast<Scalar>(i) + Scalar{0.25};
        }
        const auto result =
            simd_ops::permute<Scalar, 0>(sx::simd_unchecked_load<Simd>(input.data()), permutation);
        sx::simd_unchecked_store(result, output.data());
        for (std::size_t i = 0; i < lanes; ++i) {
            EXPECT_EQ(std::bit_cast<Bits<Scalar>>(output[i]),
                      std::bit_cast<Bits<Scalar>>(input[indices[i]]));
        }
    }
}

template <typename Scalar, std::size_t... Selectors>
void check_negation(std::index_sequence<Selectors...> selection) {
    using Simd = simd_ops::Simd<Scalar>;
    constexpr std::size_t lanes = Simd::size();
    constexpr std::array<std::size_t, lanes> selectors{Selectors...};
    constexpr auto sign_bit = Bits<Scalar>{1} << (sizeof(Scalar) * 8 - 1);
    const auto values = special_values<Scalar>();
    for (std::size_t offset = 0; offset < values.size(); ++offset) {
        std::array<Scalar, lanes> input, output;
        for (std::size_t i = 0; i < lanes; ++i) input[i] = values[(i + offset) % values.size()];
        const auto result =
            simd_ops::negate<Scalar, 0>(sx::simd_unchecked_load<Simd>(input.data()), selection);
        sx::simd_unchecked_store(result, output.data());
        for (std::size_t i = 0; i < lanes; ++i) {
            EXPECT_EQ(std::bit_cast<Bits<Scalar>>(output[i]),
                      std::bit_cast<Bits<Scalar>>(input[i]) ^ (selectors[i] ? sign_bit : 0));
        }
    }
}

template <typename Scalar>
class SimdOpsTest : public ::testing::Test {};
using Scalars = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SimdOpsTest, Scalars);

TYPED_TEST(SimdOpsTest, PermutationPreservesBits) {
    constexpr std::size_t lanes = simd_ops::Simd<TypeParam>::size();
    []<std::size_t... I>(std::index_sequence<I...>) {
        check_permutation<TypeParam>(std::index_sequence<I...>{});
        check_permutation<TypeParam>(std::index_sequence<(lanes - 1 - I)...>{});
        check_permutation<TypeParam>(std::index_sequence<((I + 1) % lanes)...>{});
        check_permutation<TypeParam>(std::index_sequence<(I % 2 % lanes)...>{});
    }(std::make_index_sequence<lanes>{});
}

TYPED_TEST(SimdOpsTest, NegationPreservesSignedZeroAndNaNPayloads) {
    constexpr std::size_t lanes = simd_ops::Simd<TypeParam>::size();
    []<std::size_t... I>(std::index_sequence<I...>) {
        check_negation<TypeParam>(std::index_sequence<(I * 0)...>{});
        check_negation<TypeParam>(std::index_sequence<(I * 0 + 1)...>{});
        check_negation<TypeParam>(std::index_sequence<(I % 2)...>{});
        check_negation<TypeParam>(std::index_sequence<((I / 2) % 2)...>{});
    }(std::make_index_sequence<lanes>{});
}
}  // namespace
