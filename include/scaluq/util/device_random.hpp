#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdint>
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

// Counter-based random numbers (Philox4x32-10, Salmon et al., SC'11).
//
// Kokkos' Random_XorShift64_Pool hands out generator states guarded by spin
// locks. When a kernel has many more work-items than the pool has states, lanes
// of one SIMD group end up waiting on each other's locks. GPUs without
// independent forward progress for such lanes (Apple GPUs through AdaptiveCpp's
// Metal backend) then never finish the kernel, and the kernel keeps running
// after the process exits. A counter-based generator needs no shared state:
// every work-item derives its numbers from (seed, index, stream). The results
// also no longer depend on the number of threads or on the backend.
//
// It is used when SCALUQ_COUNTER_BASED_RANDOM is defined, which builds with
// SCALUQ_NO_DEVICE_DOUBLE do by default. Other builds keep the Kokkos pool, so
// their seeded results do not change.
#if defined(SCALUQ_NO_DEVICE_DOUBLE) && !defined(SCALUQ_COUNTER_BASED_RANDOM)
#define SCALUQ_COUNTER_BASED_RANDOM
#endif

KOKKOS_INLINE_FUNCTION void philox4x32_10(std::uint32_t ctr[4], std::uint32_t k0, std::uint32_t k1) {
    for (int round = 0; round < 10; ++round) {
        if (round != 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        const std::uint64_t p0 = static_cast<std::uint64_t>(0xD2511F53u) * ctr[0];
        const std::uint64_t p1 = static_cast<std::uint64_t>(0xCD9E8D57u) * ctr[2];
        const std::uint32_t n0 = static_cast<std::uint32_t>(p1 >> 32) ^ ctr[1] ^ k0;
        const std::uint32_t n1 = static_cast<std::uint32_t>(p1);
        const std::uint32_t n2 = static_cast<std::uint32_t>(p0 >> 32) ^ ctr[3] ^ k1;
        const std::uint32_t n3 = static_cast<std::uint32_t>(p0);
        ctr[0] = n0;
        ctr[1] = n1;
        ctr[2] = n2;
        ctr[3] = n3;
    }
}

// A generator for one work-item. The 128-bit counter is (index, stream, block);
// block counts the Philox calls made so far, so one generator can draw any number
// of values.
class CounterGenerator {
public:
    KOKKOS_INLINE_FUNCTION CounterGenerator(std::uint64_t seed,
                                            std::uint64_t index,
                                            std::uint32_t stream)
        : _k0(static_cast<std::uint32_t>(seed)),
          _k1(static_cast<std::uint32_t>(seed >> 32)),
          _ctr{static_cast<std::uint32_t>(index),
               static_cast<std::uint32_t>(index >> 32),
               stream,
               0u} {}

    KOKKOS_INLINE_FUNCTION std::uint32_t urand() {
        if (_pos == 4) {
            for (int j = 0; j < 4; ++j) _buf[j] = _ctr[j];
            philox4x32_10(_buf, _k0, _k1);
            ++_ctr[3];
            _pos = 0;
        }
        return _buf[_pos++];
    }
    KOKKOS_INLINE_FUNCTION std::uint64_t urand64() {
        const std::uint64_t hi = urand();
        return (hi << 32) | urand();
    }
    // Used only for double precision, on devices that support double.
    KOKKOS_INLINE_FUNCTION double drand(double lo, double hi) {
        return lo + (hi - lo) * (static_cast<double>(urand64() >> 11) * 0x1p-53);
    }
    KOKKOS_INLINE_FUNCTION double normal(double mean, double stddev) {
        const double u1 = 1. - drand(0., 1.);  // (0, 1]
        const double u2 = drand(0., 1.);
        return mean + stddev * Kokkos::sqrt(-2. * Kokkos::log(u1)) *
                          Kokkos::cos(6.283185307179586 * u2);
    }

private:
    std::uint32_t _k0, _k1;
    std::uint32_t _ctr[4];
    std::uint32_t _buf[4] = {0u, 0u, 0u, 0u};
    int _pos = 4;
};

// Stream numbers keep the uses of one seed apart.
inline constexpr std::uint32_t RANDOM_STREAM_HAAR = 0;
KOKKOS_INLINE_FUNCTION constexpr std::uint32_t random_stream_sampling(std::uint32_t round) {
    // Re-sampling rounds must not reuse the numbers of earlier rounds.
    return 1u + (round << 1);
}

// The random source used by scaluq's kernels:
//   RandomSource<ExecSpace> src(seed);
//   ... auto gen = src.get(index, stream); draw from gen; src.release(gen);
// With the Kokkos pool, index and stream are ignored and get/release take and
// return a pool state.
template <typename ExecSpace>
class RandomSource {
#ifdef SCALUQ_COUNTER_BASED_RANDOM
public:
    using generator_type = CounterGenerator;
    explicit RandomSource(std::uint64_t seed) : _seed(seed) {}
    KOKKOS_INLINE_FUNCTION generator_type get(std::uint64_t index, std::uint32_t stream) const {
        return generator_type(_seed, index, stream);
    }
    KOKKOS_INLINE_FUNCTION void release(const generator_type&) const {}

private:
    std::uint64_t _seed;
#else
public:
    using pool_type = Kokkos::Random_XorShift64_Pool<ExecSpace>;
    using generator_type = typename pool_type::generator_type;
    explicit RandomSource(std::uint64_t seed) : _pool(seed) {}
    KOKKOS_INLINE_FUNCTION generator_type get(std::uint64_t, std::uint32_t) const {
        return _pool.get_state();
    }
    KOKKOS_INLINE_FUNCTION void release(const generator_type& gen) const { _pool.free_state(gen); }

private:
    pool_type _pool;
#endif
};

}  // namespace scaluq::internal
