#pragma once

#include <Kokkos_SIMD.hpp>
#include <cstddef>
#include <type_traits>

#include "../types.hpp"

namespace scaluq::internal {

template <Precision P>
struct SimdComplexTraits {
    using StorageScalar = Float<P>;
    using Scalar = std::conditional_t<P == Precision::F64, double, float>;
    using Simd = Kokkos::Experimental::simd<Scalar>;

    static constexpr std::size_t scalar_lanes = Simd::size();
    static constexpr std::size_t complex_lanes = scalar_lanes / 2;
};

template <Precision P>
class SimdComplex {
    using Traits = SimdComplexTraits<P>;
    using Scalar = typename Traits::Scalar;
    using Simd = typename Traits::Simd;
    Simd _data;

    KOKKOS_INLINE_FUNCTION static Simd splat(Scalar value) {
        return Simd(KOKKOS_LAMBDA(std::size_t) { return value; });
    }

    KOKKOS_INLINE_FUNCTION static Simd multiply_by_i(const Simd& value) {
        // We can use Memory Permutes Instructions since Kokkos 5.1
        return Simd(KOKKOS_LAMBDA(std::size_t lane) {
            const auto swapped = value[lane ^ 1ULL];
            return (lane & 1ULL) == 0 ? -swapped : swapped;
        });
    }

public:
    class Coef {
        Simd _real;
        Simd _imag;

        KOKKOS_INLINE_FUNCTION Coef(const Simd& real, const Simd& imag)
            : _real(real), _imag(imag) {}

        friend class SimdComplex;

    public:
        KOKKOS_INLINE_FUNCTION static Coef splat(const Complex<P>& coef) {
            const auto real = static_cast<Scalar>(coef.real());
            const auto imag = static_cast<Scalar>(coef.imag());
            return Coef(SimdComplex::splat(real), SimdComplex::splat(imag));
        }
    };

    static constexpr std::size_t complex_lanes = Traits::complex_lanes;
    static constexpr std::size_t scalar_lanes = Traits::scalar_lanes;

    SimdComplex() = default;
    KOKKOS_INLINE_FUNCTION explicit SimdComplex(const Simd& data) : _data(data) {}

    KOKKOS_INLINE_FUNCTION static SimdComplex load_aligned(const Complex<P>* ptr) {
        return SimdComplex(
            Simd(reinterpret_cast<const Scalar*>(ptr), Kokkos::Experimental::vector_aligned_tag{}));
    }

    KOKKOS_INLINE_FUNCTION void store_aligned(Complex<P>* ptr) const {
        _data.copy_to(reinterpret_cast<Scalar*>(ptr), Kokkos::Experimental::vector_aligned_tag{});
    }

    KOKKOS_INLINE_FUNCTION friend SimdComplex operator+(const SimdComplex& lhs,
                                                        const SimdComplex& rhs) {
        return SimdComplex(lhs._data + rhs._data);
    }

    KOKKOS_INLINE_FUNCTION friend SimdComplex operator*(const Coef& coef,
                                                        const SimdComplex& value) {
        return SimdComplex(coef._real * value._data + coef._imag * multiply_by_i(value._data));
    }

    KOKKOS_INLINE_FUNCTION friend SimdComplex operator*(const SimdComplex& value,
                                                        const Coef& coef) {
        return coef * value;
    }
};

}  // namespace scaluq::internal
