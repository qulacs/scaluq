#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

namespace scaluq::internal {

// Random numbers drawn inside device kernels.
//
// Kokkos' generators return double from drand()/normal(). Some devices have no
// double support at all (Apple GPUs through AdaptiveCpp's Metal backend), so a
// build that defines SCALUQ_NO_DEVICE_DOUBLE draws them in float instead.
// Other builds keep the double path, so seeded results do not change there.

// A float in [0, 1) built from the top 24 bits of urand(). Kokkos' frand() divides a
// 64-bit integer by 2^64 in float, which rounds up to exactly 1.0f about once in 2^25
// draws; that breaks [0, 1) for sampling and makes log(1 - u) infinite in Box-Muller.
template <typename Generator>
KOKKOS_INLINE_FUNCTION float uniform01_float(Generator& gen) {
    return static_cast<float>(gen.urand() >> 8) * 0x1p-24f;
}

template <typename FloatType, typename Generator>
KOKKOS_INLINE_FUNCTION FloatType uniform01(Generator& gen) {
#ifdef SCALUQ_NO_DEVICE_DOUBLE
    if constexpr (!std::is_same_v<FloatType, double>) {
        return static_cast<FloatType>(uniform01_float(gen));
    } else
#endif
    {
        return static_cast<FloatType>(gen.drand(0., 1.));
    }
}

template <typename FloatType, typename Generator>
KOKKOS_INLINE_FUNCTION FloatType standard_normal(Generator& gen) {
#ifdef SCALUQ_NO_DEVICE_DOUBLE
    if constexpr (!std::is_same_v<FloatType, double>) {
        // Box-Muller in float. u1 lies in (0, 1], so log() is finite.
        const float u1 = 1.f - uniform01_float(gen);
        const float u2 = uniform01_float(gen);
        const float r = Kokkos::sqrt(-2.f * Kokkos::log(u1));
        return static_cast<FloatType>(r * Kokkos::cos(6.2831853071795864f * u2));
    } else
#endif
    {
        return static_cast<FloatType>(gen.normal(0.0, 1.0));
    }
}

}  // namespace scaluq::internal
