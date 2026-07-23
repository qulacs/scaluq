#pragma once

#include <Kokkos_SIMD.hpp>
#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(KOKKOS_ARCH_AVX2)
#include <immintrin.h>
#endif

namespace scaluq::internal::simd_ops {

template <typename Scalar, std::size_t Lanes = std::is_same_v<Scalar, float> ? 8 : 0>
using Simd = Kokkos::Experimental::simd<Scalar, Lanes>;

// Reorders SIMD lanes according to the source index of each output lane.
template <typename Scalar, std::size_t Lanes, std::size_t... Indices>
KOKKOS_INLINE_FUNCTION Simd<Scalar, Lanes> permute(const Simd<Scalar, Lanes>& value,
                                            std::index_sequence<Indices...>) {
    constexpr std::size_t simd_lanes = Simd<Scalar, Lanes>::size();
    static_assert(sizeof...(Indices) == simd_lanes);
    static_assert(((Indices < simd_lanes) && ...));
#if defined(KOKKOS_ARCH_AVX2)
    if constexpr (std::is_same_v<Scalar, double> && simd_lanes == 4) {
    constexpr int control = [] {
        constexpr std::size_t bits_per_lane = std::bit_width(std::size_t{4} - 1);
        constexpr std::array<std::size_t, sizeof...(Indices)> indices{Indices...};
        int result = 0;
        for (std::size_t lane = 0; lane < indices.size(); ++lane) {
            result |= static_cast<int>(indices[lane] << (bits_per_lane * lane));
        }
        return result;
    }();
        return Simd<Scalar>(_mm256_permute4x64_pd(static_cast<__m256d>(value), control));
    } else if constexpr (std::is_same_v<Scalar, float> && simd_lanes == 8) {
        const __m256i indices = _mm256_setr_epi32(static_cast<int>(Indices)...);
        return Simd<Scalar, Lanes>(_mm256_permutevar8x32_ps(static_cast<__m256>(value), indices));
    } else if constexpr (std::is_same_v<Scalar, float> && simd_lanes == 4) {
        constexpr int control = [] {
            constexpr std::array<std::size_t, sizeof...(Indices)> indices{Indices...};
            int result = 0;
            for (std::size_t lane = 0; lane < indices.size(); ++lane) {
                result |= static_cast<int>(indices[lane] << (2 * lane));
            }
            return result;
        }();
        return Simd<Scalar, Lanes>(_mm_permute_ps(static_cast<__m128>(value), control));
    }
#endif
    constexpr std::array<std::size_t, sizeof...(Indices)> indices{Indices...};
    return Simd<Scalar, Lanes>(KOKKOS_LAMBDA(std::size_t lane) { return value[indices[lane]]; });
}

// Negates each SIMD lane whose selector is one.
template <typename Scalar, std::size_t Lanes, std::size_t... Selectors>
KOKKOS_INLINE_FUNCTION Simd<Scalar, Lanes> negate(const Simd<Scalar, Lanes>& value,
                                           std::index_sequence<Selectors...>) {
    constexpr std::size_t simd_lanes = Simd<Scalar, Lanes>::size();
    static_assert(sizeof...(Selectors) == simd_lanes);
    static_assert(((Selectors < 2) && ...));
#if defined(KOKKOS_ARCH_AVX2)
    if constexpr (std::is_same_v<Scalar, double> && simd_lanes == 4) {
        const __m256d sign = _mm256_setr_pd((Selectors == 1 ? -0.0 : 0.0)...);
        return Simd<Scalar>(_mm256_xor_pd(static_cast<__m256d>(value), sign));
    } else if constexpr (std::is_same_v<Scalar, float> && simd_lanes == 8) {
        const __m256 sign = _mm256_setr_ps((Selectors == 1 ? -0.0F : 0.0F)...);
        return Simd<Scalar, Lanes>(_mm256_xor_ps(static_cast<__m256>(value), sign));
    } else if constexpr (std::is_same_v<Scalar, float> && simd_lanes == 4) {
        const __m128 sign = _mm_setr_ps((Selectors == 1 ? -0.0F : 0.0F)...);
        return Simd<Scalar, Lanes>(_mm_xor_ps(static_cast<__m128>(value), sign));
    }
#endif
    constexpr std::array<std::size_t, sizeof...(Selectors)> selectors{Selectors...};
    return Simd<Scalar, Lanes>(KOKKOS_LAMBDA(std::size_t lane) {
        return selectors[lane] == 1 ? -value[lane] : value[lane];
    });
}

}  // namespace scaluq::internal::simd_ops
