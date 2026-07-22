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

    friend class Coef;

public:
    class Coef {
        Simd _real, _imag;
        KOKKOS_INLINE_FUNCTION Coef(Scalar real, Scalar imag) : _real(real), _imag(imag) {}
        friend class SimdComplex;

    public:
        KOKKOS_INLINE_FUNCTION static Coef splat(const Complex<P>& coef) {
            return Coef(static_cast<Scalar>(coef.real()), static_cast<Scalar>(coef.imag()));
        }

        KOKKOS_INLINE_FUNCTION SimdComplex operator*(const SimdComplex& value) const {
            return SimdComplex(_real * value._data + _imag * value.multiply_by_i()._data);
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

    KOKKOS_INLINE_FUNCTION SimdComplex multiply_by_i() const {
        // TODO: もっと効率的にできそう
        return SimdComplex(Simd(KOKKOS_LAMBDA(std::size_t lane) {
            const auto swapped = _data[lane ^ 1ULL];
            return (lane & 1ULL) == 0 ? -swapped : swapped;
        }));
    }

    KOKKOS_INLINE_FUNCTION friend SimdComplex operator+(const SimdComplex& lhs,
                                                        const SimdComplex& rhs) {
        return SimdComplex(lhs._data + rhs._data);
    }
};

}  // namespace scaluq::internal
