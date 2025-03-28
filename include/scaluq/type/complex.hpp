#pragma once

#include <Kokkos_Core.hpp>
#include <complex>

#include "floating_point.hpp"

namespace scaluq::internal {
template <Precision Prec>
class Complex {
    using FloatType = Float<Prec>;

public:
    KOKKOS_INLINE_FUNCTION Complex() : _real{0}, _imag{0} {}
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION Complex(Scalar real)
        : _real(static_cast<FloatType>(real)), _imag{0} {}
    template <typename Scalar1, typename Scalar2>
    requires(std::is_same_v<Scalar1, FloatType> || std::is_same_v<Scalar1, double> ||
             std::is_same_v<Scalar1, int>) &&
        (std::is_same_v<Scalar2, FloatType> || std::is_same_v<Scalar2, double> ||
         std::is_same_v<Scalar2, int>)KOKKOS_INLINE_FUNCTION Complex(Scalar1 real, Scalar2 imag)
        : _real(static_cast<FloatType>(real)),
    _imag(static_cast<FloatType>(imag)) {}
    KOKKOS_INLINE_FUNCTION Complex(const Complex& other)
        : _real(other.real()), _imag(other.imag()) {}
    KOKKOS_INLINE_FUNCTION Complex(const std::complex<double>& c)
        : _real(static_cast<FloatType>(c.real())), _imag(static_cast<FloatType>(c.imag())) {}

    KOKKOS_INLINE_FUNCTION Complex& operator=(const Complex& other) {
        _real = other._real;
        _imag = other._imag;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION Complex& operator=(const std::complex<double>& c) {
        _real = static_cast<FloatType>(c.real());
        _imag = static_cast<FloatType>(c.imag());
        return *this;
    }
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
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
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION Complex& operator*=(Scalar rhs) {
        _real *= static_cast<FloatType>(rhs);
        _imag *= static_cast<FloatType>(rhs);
        return *this;
    }
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION Complex operator*(Scalar rhs) const {
        return Complex(*this) *= static_cast<FloatType>(rhs);
    }
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION friend Complex operator*(Scalar lhs, const Complex& rhs) {
        return Complex(static_cast<FloatType>(lhs) * rhs._real,
                       static_cast<FloatType>(lhs) * rhs._imag);
    }
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION Complex& operator/=(Scalar rhs) {
        _real /= static_cast<FloatType>(rhs);
        _imag /= static_cast<FloatType>(rhs);
        return *this;
    }
    template <typename Scalar>
    requires std::is_same_v<Scalar, FloatType> || std::is_same_v<Scalar, double> ||
        std::is_same_v<Scalar, int>
            KOKKOS_INLINE_FUNCTION Complex operator/(Scalar rhs) const {
        return Complex(*this) /= static_cast<FloatType>(rhs);
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
}  // namespace scaluq::internal
