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
template <typename FloatType, typename Generator>
KOKKOS_INLINE_FUNCTION FloatType uniform01(Generator& gen) {
#ifdef SCALUQ_NO_DEVICE_DOUBLE
    if constexpr (!std::is_same_v<FloatType, double>) {
        return static_cast<FloatType>(gen.frand(0.f, 1.f));
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
        // Box-Muller in float. 1 - frand() lies in (0, 1], so log() is finite.
        const float u1 = 1.f - gen.frand();
        const float u2 = gen.frand();
        const float r = Kokkos::sqrt(-2.f * Kokkos::log(u1));
        return static_cast<FloatType>(r * Kokkos::cos(6.2831853071795864f * u2));
    } else
#endif
    {
        return static_cast<FloatType>(gen.normal(0.0, 1.0));
    }
}

}  // namespace scaluq::internal
