#pragma once

#include <Kokkos_SIMD.hpp>
#include <array>
#include <bit>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(KOKKOS_ARCH_AVX2) || defined(KOKKOS_ARCH_AVX512XEON)
#include <immintrin.h>
#endif

namespace scaluq::internal::simd_ops {

template <typename Scalar>
using Simd = Kokkos::Experimental::simd<Scalar>;

// Reorders SIMD lanes according to the source index of each output lane.
template <typename Scalar, std::size_t... Indices>
KOKKOS_INLINE_FUNCTION Simd<Scalar> permute(const Simd<Scalar>& value,
                                            std::index_sequence<Indices...>) {
    static_assert(sizeof...(Indices) == Simd<Scalar>::size());
    static_assert(((Indices < Simd<Scalar>::size()) && ...));
#if defined(KOKKOS_ARCH_AVX2)
    constexpr int control = [] {
        constexpr std::size_t bits_per_lane = std::bit_width(std::size_t{4} - 1);
        constexpr std::array<std::size_t, sizeof...(Indices)> indices{Indices...};
        int result = 0;
        for (std::size_t lane = 0; lane < indices.size(); ++lane) {
            result |= static_cast<int>(indices[lane] << (bits_per_lane * lane));
        }
        return result;
    }();
    if constexpr (std::is_same_v<Scalar, double>) {
        return Simd<Scalar>(_mm256_permute4x64_pd(static_cast<__m256d>(value), control));
    } else if constexpr (std::is_same_v<Scalar, float>) {
        return Simd<Scalar>(_mm_permute_ps(static_cast<__m128>(value), control));
    }
#elif defined(KOKKOS_ARCH_AVX512XEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m512i indices = _mm512_setr_epi64(static_cast<long long>(Indices)...);
        return Simd<Scalar>(_mm512_permutexvar_pd(indices, static_cast<__m512d>(value)));
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m256i indices = _mm256_setr_epi32(static_cast<int>(Indices)...);
        return Simd<Scalar>(_mm256_permutexvar_ps(indices, static_cast<__m256>(value)));
    }
#endif
    constexpr std::array<std::size_t, sizeof...(Indices)> indices{Indices...};
    return Simd<Scalar>(KOKKOS_LAMBDA(std::size_t lane) { return value[indices[lane]]; });
}

// Negates each SIMD lane whose selector is one.
template <typename Scalar, std::size_t... Selectors>
KOKKOS_INLINE_FUNCTION Simd<Scalar> negate(const Simd<Scalar>& value,
                                           std::index_sequence<Selectors...>) {
    static_assert(sizeof...(Selectors) == Simd<Scalar>::size());
    static_assert(((Selectors < 2) && ...));
#if defined(KOKKOS_ARCH_AVX2)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m256d sign = _mm256_setr_pd((Selectors == 1 ? -0.0 : 0.0)...);
        return Simd<Scalar>(_mm256_xor_pd(static_cast<__m256d>(value), sign));
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m128 sign = _mm_setr_ps((Selectors == 1 ? -0.0F : 0.0F)...);
        return Simd<Scalar>(_mm_xor_ps(static_cast<__m128>(value), sign));
    }
#elif defined(KOKKOS_ARCH_AVX512XEON)
    if constexpr (std::is_same_v<Scalar, double>) {
        const __m512d sign = _mm512_setr_pd((Selectors == 1 ? -0.0 : 0.0)...);
        return Simd<Scalar>(_mm512_xor_pd(static_cast<__m512d>(value), sign));
    } else if constexpr (std::is_same_v<Scalar, float>) {
        const __m256 sign = _mm256_setr_ps((Selectors == 1 ? -0.0F : 0.0F)...);
        return Simd<Scalar>(_mm256_xor_ps(static_cast<__m256>(value), sign));
    }
#endif
    constexpr std::array<std::size_t, sizeof...(Selectors)> selectors{Selectors...};
    return Simd<Scalar>(KOKKOS_LAMBDA(std::size_t lane) {
        return selectors[lane] == 1 ? -value[lane] : value[lane];
    });
}

}  // namespace scaluq::internal::simd_ops
