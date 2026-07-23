#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "../util/simd.hpp"
#include "floating_point.hpp"

namespace scaluq::internal {
enum class CoefKind { General, Real, Imag, Zero, One };

constexpr CoefKind common_coef_kind(CoefKind lhs, CoefKind rhs) {
    if (lhs == CoefKind::Zero && rhs == CoefKind::Zero) return CoefKind::Zero;
    if (lhs == CoefKind::One && rhs == CoefKind::One) return CoefKind::One;
    const bool lhs_real = lhs == CoefKind::Real || lhs == CoefKind::Zero || lhs == CoefKind::One;
    const bool rhs_real = rhs == CoefKind::Real || rhs == CoefKind::Zero || rhs == CoefKind::One;
    if (lhs_real && rhs_real) return CoefKind::Real;
    const bool lhs_imag = lhs == CoefKind::Imag || lhs == CoefKind::Zero;
    const bool rhs_imag = rhs == CoefKind::Imag || rhs == CoefKind::Zero;
    if (lhs_imag && rhs_imag) return CoefKind::Imag;
    return CoefKind::General;
}

constexpr CoefKind common_coef_kind(CoefKind a, CoefKind b, CoefKind c, CoefKind d) {
    return common_coef_kind(common_coef_kind(a, b), common_coef_kind(c, d));
}

template <Precision Prec>
class Complex {
    using FloatType = Float<Prec>;

public:
    class Coef {
        FloatType _real, _imag;

        KOKKOS_INLINE_FUNCTION Coef(FloatType real, FloatType imag) : _real(real), _imag(imag) {}

    public:
        KOKKOS_INLINE_FUNCTION static Coef splat(const Complex& coef) {
            return Coef(coef.real(), coef.imag());
        }
        KOKKOS_INLINE_FUNCTION Complex operator*(const Complex& value) const {
            return Complex(_real * value.real() - _imag * value.imag(),
                           _real * value.imag() + _imag * value.real());
        }
    };

    class RCoef {
        FloatType _value;
        KOKKOS_INLINE_FUNCTION explicit RCoef(FloatType value) : _value(value) {}

    public:
        KOKKOS_INLINE_FUNCTION static RCoef splat(const Complex& coef) {
            return RCoef(coef.real());
        }
        KOKKOS_INLINE_FUNCTION Complex operator*(const Complex& value) const {
            return Complex(_value * value.real(), _value * value.imag());
        }
    };

    class ICoef {
        FloatType _value;
        KOKKOS_INLINE_FUNCTION explicit ICoef(FloatType value) : _value(value) {}

    public:
        KOKKOS_INLINE_FUNCTION static ICoef splat(const Complex& coef) {
            return ICoef(coef.imag());
        }
        KOKKOS_INLINE_FUNCTION Complex operator*(const Complex& value) const {
            return Complex(-_value * value.imag(), _value * value.real());
        }
    };

    class ZeroExpression {
    public:
        KOKKOS_INLINE_FUNCTION operator Complex() const { return Complex{}; }
    };

    class ZeroCoef {
    public:
        KOKKOS_INLINE_FUNCTION static ZeroCoef splat(const Complex&) { return {}; }
        KOKKOS_INLINE_FUNCTION ZeroExpression operator*(const Complex&) const { return {}; }
    };

    class OneCoef {
    public:
        KOKKOS_INLINE_FUNCTION static OneCoef splat(const Complex&) { return {}; }
        KOKKOS_INLINE_FUNCTION Complex operator*(const Complex& value) const { return value; }
    };

    KOKKOS_INLINE_FUNCTION Complex() : _real{0}, _imag{0} {}
    KOKKOS_INLINE_FUNCTION Complex(ZeroExpression) : Complex() {}
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex(Scalar real) : _real(static_cast<FloatType>(real)), _imag{0} {}
    template <typename Scalar1, typename Scalar2>
    KOKKOS_INLINE_FUNCTION Complex(Scalar1 real, Scalar2 imag)
        : _real(static_cast<FloatType>(real)), _imag(static_cast<FloatType>(imag)) {}
    KOKKOS_INLINE_FUNCTION Complex(const std::complex<double>& c)
        : _real(static_cast<FloatType>(c.real())), _imag(static_cast<FloatType>(c.imag())) {}

    KOKKOS_INLINE_FUNCTION Complex& operator=(const std::complex<double>& c) {
        _real = static_cast<FloatType>(c.real());
        _imag = static_cast<FloatType>(c.imag());
        return *this;
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex& operator=(Scalar real) {
        _real = static_cast<FloatType>(real);
        _imag = FloatType{0};
        return *this;
    }

    KOKKOS_INLINE_FUNCTION operator std::complex<double>() const {
        return std::complex(static_cast<double>(_real), static_cast<double>(_imag));
    }

    KOKKOS_INLINE_FUNCTION const FloatType& real() const { return _real; };
    KOKKOS_INLINE_FUNCTION FloatType& real() { return _real; };
    KOKKOS_INLINE_FUNCTION const FloatType& imag() const { return _imag; };
    KOKKOS_INLINE_FUNCTION FloatType& imag() { return _imag; };

    KOKKOS_INLINE_FUNCTION Complex operator+() const { return *this; }
    KOKKOS_INLINE_FUNCTION Complex operator-() const { return Complex(-_real, -_imag); }

    KOKKOS_INLINE_FUNCTION Complex& operator+=(const Complex& rhs) {
        _real += rhs._real;
        _imag += rhs._imag;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION Complex operator+(const Complex& rhs) const {
        return Complex(*this) += rhs;
    }
    KOKKOS_INLINE_FUNCTION friend Complex operator+(ZeroExpression, const Complex& rhs) {
        return rhs;
    }
    KOKKOS_INLINE_FUNCTION friend Complex operator+(const Complex& lhs, ZeroExpression) {
        return lhs;
    }
    KOKKOS_INLINE_FUNCTION friend ZeroExpression operator+(ZeroExpression, ZeroExpression) {
        return {};
    }
    KOKKOS_INLINE_FUNCTION Complex& operator-=(const Complex& rhs) {
        _real -= rhs._real;
        _imag -= rhs._imag;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION Complex operator-(const Complex& rhs) const {
        return Complex(*this) -= rhs;
    }
    KOKKOS_INLINE_FUNCTION Complex operator*(const Complex& rhs) const {
        return Complex(_real * rhs._real - _imag * rhs._imag,
                       _real * rhs._imag + _imag * rhs._real);
    }
    KOKKOS_INLINE_FUNCTION Complex& operator*=(const Complex& rhs) { return *this = *this * rhs; }
    KOKKOS_INLINE_FUNCTION Complex& operator*=(const std::complex<double>& rhs) {
        return *this *= Complex(rhs);
    }
    KOKKOS_INLINE_FUNCTION Complex operator*(const std::complex<double>& rhs) const {
        return *this * Complex(rhs);
    }
    KOKKOS_INLINE_FUNCTION friend Complex operator*(const std::complex<double>& lhs,
                                                     const Complex& rhs) {
        return Complex(lhs) * rhs;
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex& operator*=(Scalar rhs) {
        _real *= static_cast<FloatType>(rhs);
        _imag *= static_cast<FloatType>(rhs);
        return *this;
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex operator*(Scalar rhs) const {
        return Complex(*this) *= static_cast<FloatType>(rhs);
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION friend Complex operator*(Scalar lhs, const Complex& rhs) {
        return Complex(static_cast<FloatType>(lhs) * rhs._real,
                       static_cast<FloatType>(lhs) * rhs._imag);
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex& operator/=(Scalar rhs) {
        _real /= static_cast<FloatType>(rhs);
        _imag /= static_cast<FloatType>(rhs);
        return *this;
    }
    template <typename Scalar>
    KOKKOS_INLINE_FUNCTION Complex operator/(Scalar rhs) const {
        return Complex(*this) /= static_cast<FloatType>(rhs);
    }
    KOKKOS_INLINE_FUNCTION Complex& operator/=(const Complex& rhs) {
        FloatType denominator = rhs._real * rhs._real + rhs._imag * rhs._imag;
        FloatType real_part = _real * rhs._real + _imag * rhs._imag;
        FloatType imag_part = _imag * rhs._real - _real * rhs._imag;
        _real = real_part / denominator;
        _imag = imag_part / denominator;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION Complex operator/(const Complex& rhs) const {
        return Complex(*this) /= rhs;
    }
    KOKKOS_INLINE_FUNCTION Complex& operator/=(const std::complex<double>& rhs) {
        return *this /= Complex(rhs);
    }
    KOKKOS_INLINE_FUNCTION Complex operator/(const std::complex<double>& rhs) const {
        return *this / Complex(rhs);
    }

private:
    FloatType _real, _imag;
};

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Complex<Prec> conj(const Complex<Prec>& c) {
    return Complex<Prec>(c.real(), -c.imag());
}
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Float<Prec> real(const Complex<Prec>& c) {
    return c.real();
}
template <Precision Prec>
KOKKOS_INLINE_FUNCTION Float<Prec> imag(const Complex<Prec>& c) {
    return c.imag();
}

namespace simd_complex_detail {

template <std::size_t Mask, std::size_t... Lanes>
auto make_xor_permutation(std::index_sequence<Lanes...>) -> std::index_sequence<(Lanes ^ Mask)...>;

template <std::size_t Mask, std::size_t LaneCount>
using XorPermutation = decltype(make_xor_permutation<Mask>(std::make_index_sequence<LaneCount>{}));

template <std::size_t... Lanes>
auto make_even_lane_selection(std::index_sequence<Lanes...>)
    -> std::index_sequence<static_cast<std::size_t>((Lanes & 1ULL) == 0)...>;

template <std::size_t LaneCount>
using EvenLaneSelection = decltype(make_even_lane_selection(std::make_index_sequence<LaneCount>{}));

}  // namespace simd_complex_detail

template <Precision P, std::size_t ScalarLanes = P == Precision::F32 ? 8 : 4>
class SimdComplex {
    static_assert(P == Precision::F32 || P == Precision::F64);

    using Scalar = std::conditional_t<P == Precision::F64, double, float>;
    using Simd = simd_ops::Simd<Scalar, ScalarLanes>;
    Simd _data;

public:
    class Coef {
        Simd _real, _imag;
        KOKKOS_INLINE_FUNCTION Coef(Scalar real, Scalar imag) : _real(real), _imag(imag) {}
        KOKKOS_INLINE_FUNCTION Coef(const Simd& real, const Simd& imag)
            : _real(real), _imag(imag) {}

    public:
        KOKKOS_INLINE_FUNCTION static Coef splat(const Complex<P>& coef) {
            return Coef(static_cast<Scalar>(coef.real()), static_cast<Scalar>(coef.imag()));
        }

        template <std::size_t ComplexLaneBit>
        KOKKOS_INLINE_FUNCTION static Coef select_complex_lane_bit(const Complex<P>& zero,
                                                                   const Complex<P>& one) {
            constexpr std::size_t complex_lane_mask = 1ULL << ComplexLaneBit;
            static_assert(complex_lane_mask < complex_lanes);
            const Scalar zero_real = static_cast<Scalar>(zero.real());
            const Scalar zero_imag = static_cast<Scalar>(zero.imag());
            const Scalar one_real = static_cast<Scalar>(one.real());
            const Scalar one_imag = static_cast<Scalar>(one.imag());
            return Coef(Simd(KOKKOS_LAMBDA(std::size_t lane) {
                            return ((lane >> 1) & complex_lane_mask) == 0 ? zero_real : one_real;
                        }),
                        Simd(KOKKOS_LAMBDA(std::size_t lane) {
                            return ((lane >> 1) & complex_lane_mask) == 0 ? zero_imag : one_imag;
                        }));
        }

        template <std::size_t ComplexLaneBit0, std::size_t ComplexLaneBit1>
        KOKKOS_INLINE_FUNCTION static Coef select_complex_lane_bits(const Complex<P>& zero_zero,
                                                                    const Complex<P>& zero_one,
                                                                    const Complex<P>& one_zero,
                                                                    const Complex<P>& one_one) {
            constexpr std::size_t complex_lane_mask0 = 1ULL << ComplexLaneBit0;
            constexpr std::size_t complex_lane_mask1 = 1ULL << ComplexLaneBit1;
            static_assert(ComplexLaneBit0 != ComplexLaneBit1);
            static_assert(complex_lane_mask0 < complex_lanes);
            static_assert(complex_lane_mask1 < complex_lanes);
            const Scalar real[4] = {static_cast<Scalar>(zero_zero.real()),
                                    static_cast<Scalar>(zero_one.real()),
                                    static_cast<Scalar>(one_zero.real()),
                                    static_cast<Scalar>(one_one.real())};
            const Scalar imag[4] = {static_cast<Scalar>(zero_zero.imag()),
                                    static_cast<Scalar>(zero_one.imag()),
                                    static_cast<Scalar>(one_zero.imag()),
                                    static_cast<Scalar>(one_one.imag())};
            return Coef(
                Simd(KOKKOS_LAMBDA(std::size_t lane) {
                    const std::size_t complex_lane = lane >> 1;
                    const std::size_t index = ((complex_lane & complex_lane_mask0) != 0 ? 1 : 0) |
                                              ((complex_lane & complex_lane_mask1) != 0 ? 2 : 0);
                    return real[index];
                }),
                Simd(KOKKOS_LAMBDA(std::size_t lane) {
                    const std::size_t complex_lane = lane >> 1;
                    const std::size_t index = ((complex_lane & complex_lane_mask0) != 0 ? 1 : 0) |
                                              ((complex_lane & complex_lane_mask1) != 0 ? 2 : 0);
                    return imag[index];
                }));
        }

        KOKKOS_INLINE_FUNCTION SimdComplex operator*(const SimdComplex& value) const {
            return SimdComplex(_real * value._data + _imag * value.multiply_by_i()._data);
        }
    };

    class RCoef {
        Simd _value;
        KOKKOS_INLINE_FUNCTION explicit RCoef(Scalar value) : _value(value) {}
        KOKKOS_INLINE_FUNCTION explicit RCoef(const Simd& value) : _value(value) {}

    public:
        KOKKOS_INLINE_FUNCTION static RCoef splat(const Complex<P>& coef) {
            return RCoef(static_cast<Scalar>(coef.real()));
        }

        template <std::size_t ComplexLaneBit>
        KOKKOS_INLINE_FUNCTION static RCoef select_complex_lane_bit(const Complex<P>& zero,
                                                                    const Complex<P>& one) {
            constexpr std::size_t complex_lane_mask = 1ULL << ComplexLaneBit;
            static_assert(complex_lane_mask < complex_lanes);
            const Scalar zero_real = static_cast<Scalar>(zero.real());
            const Scalar one_real = static_cast<Scalar>(one.real());
            return RCoef(Simd(KOKKOS_LAMBDA(std::size_t lane) {
                return ((lane >> 1) & complex_lane_mask) == 0 ? zero_real : one_real;
            }));
        }

        template <std::size_t ComplexLaneBit0, std::size_t ComplexLaneBit1>
        KOKKOS_INLINE_FUNCTION static RCoef select_complex_lane_bits(const Complex<P>& c00,
                                                                     const Complex<P>& c01,
                                                                     const Complex<P>& c10,
                                                                     const Complex<P>& c11) {
            constexpr std::size_t mask0 = 1ULL << ComplexLaneBit0;
            constexpr std::size_t mask1 = 1ULL << ComplexLaneBit1;
            static_assert(ComplexLaneBit0 != ComplexLaneBit1);
            static_assert(mask0 < complex_lanes);
            static_assert(mask1 < complex_lanes);
            const Scalar values[4] = {static_cast<Scalar>(c00.real()),
                                      static_cast<Scalar>(c01.real()),
                                      static_cast<Scalar>(c10.real()),
                                      static_cast<Scalar>(c11.real())};
            return RCoef(Simd(KOKKOS_LAMBDA(std::size_t lane) {
                const std::size_t complex_lane = lane >> 1;
                const std::size_t index =
                    ((complex_lane & mask0) != 0 ? 1 : 0) | ((complex_lane & mask1) != 0 ? 2 : 0);
                return values[index];
            }));
        }

        KOKKOS_INLINE_FUNCTION SimdComplex operator*(const SimdComplex& value) const {
            return SimdComplex(_value * value._data);
        }
    };

    class ICoef {
        Simd _value;
        KOKKOS_INLINE_FUNCTION explicit ICoef(Scalar value) : _value(value) {}
        KOKKOS_INLINE_FUNCTION explicit ICoef(const Simd& value) : _value(value) {}

    public:
        KOKKOS_INLINE_FUNCTION static ICoef splat(const Complex<P>& coef) {
            return ICoef(static_cast<Scalar>(coef.imag()));
        }

        template <std::size_t ComplexLaneBit>
        KOKKOS_INLINE_FUNCTION static ICoef select_complex_lane_bit(const Complex<P>& zero,
                                                                    const Complex<P>& one) {
            constexpr std::size_t complex_lane_mask = 1ULL << ComplexLaneBit;
            static_assert(complex_lane_mask < complex_lanes);
            const Scalar zero_imag = static_cast<Scalar>(zero.imag());
            const Scalar one_imag = static_cast<Scalar>(one.imag());
            return ICoef(Simd(KOKKOS_LAMBDA(std::size_t lane) {
                return ((lane >> 1) & complex_lane_mask) == 0 ? zero_imag : one_imag;
            }));
        }

        template <std::size_t ComplexLaneBit0, std::size_t ComplexLaneBit1>
        KOKKOS_INLINE_FUNCTION static ICoef select_complex_lane_bits(const Complex<P>& c00,
                                                                     const Complex<P>& c01,
                                                                     const Complex<P>& c10,
                                                                     const Complex<P>& c11) {
            constexpr std::size_t mask0 = 1ULL << ComplexLaneBit0;
            constexpr std::size_t mask1 = 1ULL << ComplexLaneBit1;
            static_assert(ComplexLaneBit0 != ComplexLaneBit1);
            static_assert(mask0 < complex_lanes);
            static_assert(mask1 < complex_lanes);
            const Scalar values[4] = {static_cast<Scalar>(c00.imag()),
                                      static_cast<Scalar>(c01.imag()),
                                      static_cast<Scalar>(c10.imag()),
                                      static_cast<Scalar>(c11.imag())};
            return ICoef(Simd(KOKKOS_LAMBDA(std::size_t lane) {
                const std::size_t complex_lane = lane >> 1;
                const std::size_t index =
                    ((complex_lane & mask0) != 0 ? 1 : 0) | ((complex_lane & mask1) != 0 ? 2 : 0);
                return values[index];
            }));
        }

        KOKKOS_INLINE_FUNCTION SimdComplex operator*(const SimdComplex& value) const {
            return SimdComplex(_value * value.multiply_by_i()._data);
        }
    };

    class ZeroExpression {
    public:
        // 項がすべて ZeroExpression のときに呼ばれうる
        KOKKOS_INLINE_FUNCTION void store_aligned(Complex<P>* ptr) const {
            Simd(Scalar{0}).copy_to(reinterpret_cast<Scalar*>(ptr),
                                    Kokkos::Experimental::vector_aligned_tag{});
        }
    };

    class ZeroCoef {
    public:
        KOKKOS_INLINE_FUNCTION static ZeroCoef splat(const Complex<P>&) { return {}; }
        template <std::size_t ComplexLaneBit>
        KOKKOS_INLINE_FUNCTION static ZeroCoef select_complex_lane_bit(const Complex<P>&,
                                                                       const Complex<P>&) {
            return {};
        }
        template <std::size_t ComplexLaneBit0, std::size_t ComplexLaneBit1>
        KOKKOS_INLINE_FUNCTION static ZeroCoef select_complex_lane_bits(const Complex<P>&,
                                                                        const Complex<P>&,
                                                                        const Complex<P>&,
                                                                        const Complex<P>&) {
            return {};
        }
        KOKKOS_INLINE_FUNCTION ZeroExpression operator*(const SimdComplex&) const { return {}; }
    };

    class OneCoef {
    public:
        KOKKOS_INLINE_FUNCTION static OneCoef splat(const Complex<P>&) { return {}; }
        template <std::size_t ComplexLaneBit>
        KOKKOS_INLINE_FUNCTION static OneCoef select_complex_lane_bit(const Complex<P>&,
                                                                      const Complex<P>&) {
            return {};
        }
        template <std::size_t ComplexLaneBit0, std::size_t ComplexLaneBit1>
        KOKKOS_INLINE_FUNCTION static OneCoef select_complex_lane_bits(const Complex<P>&,
                                                                       const Complex<P>&,
                                                                       const Complex<P>&,
                                                                       const Complex<P>&) {
            return {};
        }
        KOKKOS_INLINE_FUNCTION SimdComplex operator*(const SimdComplex& value) const {
            return value;
        }
    };

    static constexpr std::size_t scalar_lanes = Simd::size();
    static constexpr std::size_t complex_lanes = scalar_lanes / 2;

    KOKKOS_INLINE_FUNCTION explicit SimdComplex(const Simd& data) : _data(data) {}

    KOKKOS_INLINE_FUNCTION static SimdComplex load_aligned(const Complex<P>* ptr) {
        return SimdComplex(
            Simd(reinterpret_cast<const Scalar*>(ptr), Kokkos::Experimental::vector_aligned_tag{}));
    }

    KOKKOS_INLINE_FUNCTION void store_aligned(Complex<P>* ptr) const {
        _data.copy_to(reinterpret_cast<Scalar*>(ptr), Kokkos::Experimental::vector_aligned_tag{});
    }

    KOKKOS_INLINE_FUNCTION SimdComplex multiply_by_i() const {
        using Permutation = simd_complex_detail::XorPermutation<1, scalar_lanes>;
        using NegatedLanes = simd_complex_detail::EvenLaneSelection<scalar_lanes>;
        return SimdComplex(simd_ops::negate<Scalar, ScalarLanes>(
            simd_ops::permute<Scalar, ScalarLanes>(_data, Permutation{}), NegatedLanes{}));
    }

    template <std::size_t ComplexLaneMask>
    KOKKOS_INLINE_FUNCTION SimdComplex permute_complex_lanes_xor() const {
        static_assert(ComplexLaneMask != 0);
        static_assert(ComplexLaneMask < complex_lanes);
        using Permutation =
            simd_complex_detail::XorPermutation<(ComplexLaneMask << 1), scalar_lanes>;
        return SimdComplex(simd_ops::permute<Scalar, ScalarLanes>(_data, Permutation{}));
    }

    KOKKOS_INLINE_FUNCTION friend SimdComplex operator+(const SimdComplex& lhs,
                                                        const SimdComplex& rhs) {
        return SimdComplex(lhs._data + rhs._data);
    }

    // ゼロの項に対する加算を削減
    KOKKOS_INLINE_FUNCTION friend SimdComplex operator+(ZeroExpression, const SimdComplex& rhs) {
        return rhs;
    }
    KOKKOS_INLINE_FUNCTION friend SimdComplex operator+(const SimdComplex& lhs, ZeroExpression) {
        return lhs;
    }
    KOKKOS_INLINE_FUNCTION friend ZeroExpression operator+(ZeroExpression, ZeroExpression) {
        return {};
    }
};

template <typename Value, CoefKind Kind>
struct CoefType;

template <typename Value>
struct CoefType<Value, CoefKind::General> {
    using type = typename Value::Coef;
};

template <typename Value>
struct CoefType<Value, CoefKind::Real> {
    using type = typename Value::RCoef;
};

template <typename Value>
struct CoefType<Value, CoefKind::Imag> {
    using type = typename Value::ICoef;
};

template <typename Value>
struct CoefType<Value, CoefKind::Zero> {
    using type = typename Value::ZeroCoef;
};

template <typename Value>
struct CoefType<Value, CoefKind::One> {
    using type = typename Value::OneCoef;
};

template <Precision P, CoefKind Kind>
using ScalarCoef = typename CoefType<Complex<P>, Kind>::type;

template <Precision P, CoefKind Kind>
using SimdCoef = typename CoefType<SimdComplex<P>, Kind>::type;

template <typename SimdType, CoefKind Kind>
using SimdCoefFor = typename CoefType<SimdType, Kind>::type;

}  // namespace scaluq::internal
