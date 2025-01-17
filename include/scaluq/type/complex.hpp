#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <nlohmann/json.hpp>

#include "floating_point.hpp"

namespace scaluq::internal {
template <Precision Prec>
class Complex {
    using FloatType = Float<Prec>;

public:
    Complex() : _real{0}, _imag{0} {}
    template <Precision Prec2>
    Complex(Float<Prec2> real, Float<Prec2> imag = {0})
        : _real(static_cast<FloatType>(real)), _imag(static_cast<FloatType>(imag)) {}
    Complex(double real, double imag = 0.)
        : _real(static_cast<FloatType>(real)), _imag(static_cast<FloatType>(imag)) {}
    template <Precision Prec2>
    Complex(const Complex<Prec2>& other)
        : _real(static_cast<FloatType>(other._real)), _imag(static_cast<FloatType>(other._imag)) {}
    Complex(const std::complex<double>& c)
        : _real(static_cast<FloatType>(c.real())), _imag(static_cast<FloatType>(c.imag())) {}

    const FloatType& real() const { return _real; };
    FloatType& real() { return _real; };
    const FloatType& imag() const { return _imag; };
    FloatType& imag() { return _imag; };

    Complex operator+() const { return *this; }
    Complex operator-() const { return Complex(-_real, _imag); }

    Complex& operator+=(const Complex& rhs) {
        _real += rhs._real;
        _imag += rhs._imag;
        return *this;
    }
    Complex operator+(const Complex& rhs) const { return Complex(*this) += rhs; }
    Complex& operator-=(const Complex& rhs) {
        _real -= rhs._real;
        _imag -= rhs._imag;
        return *this;
    }
    Complex operator-(const Complex& rhs) const { return Complex(*this) -= rhs; }
    Complex operator*(const Complex& rhs) const {
        return Complex(_real * rhs._real - _imag * rhs._imag,
                       _real * rhs._imag + _imag * rhs._real);
    }
    Complex& operator*=(const Complex& rhs) { return *this = *this * rhs; }
    Complex& operator*=(const FloatType& rhs) {
        _real *= rhs;
        _imag *= rhs;
        return *this;
    }
    Complex operator*(const FloatType& rhs) const { return Complex(*this) *= rhs; }
    Complex& operator*=(const double& rhs) {
        _real *= static_cast<FloatType>(rhs);
        _imag *= static_cast<FloatType>(rhs);
        return *this;
    }
    Complex operator*(const double& rhs) const { return Complex(*this) *= rhs; }
    friend Complex operator*(const FloatType& lhs, const Complex& rhs) const {
        return Complex(lhs * rhs._real, lhs * rhs._imag);
    }
    friend Complex operator*(const double& lhs, const Complex& rhs) const {
        return Complex(static_cast<FloatType>(lhs) * rhs._real,
                       static_cast<FloatType>(lhs) * rhs._imag);
    }
    Complex& operator/=(const FloatType& rhs) {
        _real /= rhs;
        _imag /= rhs;
        return *this;
    }
    Complex operator/(const FloatType& rhs) const { return Complex(*this) /= rhs; }
    Complex& operator/=(const double& rhs) {
        _real /= static_cast<FloatType>(rhs);
        _imag /= static_cast<FloatType>(rhs);
        return *this;
    }
    Complex operator/(const double& rhs) const { return Complex(*this) /= rhs; }

private:
    FloatType _real, _imag;
};

}  // namespace scaluq::internal
