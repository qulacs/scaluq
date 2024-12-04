#pragma once

#include <cmath>
#include <scaluq/types.hpp>

#include "template.hpp"

#define IMPL_MATH_UNARY_FUNCTION(FUNC, FLOAT) \
    KOKKOS_INLINE_FUNCTION FLOAT FUNC(FLOAT x) { return std::FUNC(x); }

#define IMPL_ALL_MATH_UNARY_FUNCTION(FLOAT) \
    IMPL_MATH_UNARY_FUNCTION(sqrt, FLOAT)   \
    IMPL_MATH_UNARY_FUNCTION(sin, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(cos, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(tan, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(sinh, FLOAT)   \
    IMPL_MATH_UNARY_FUNCTION(cosh, FLOAT)   \
    IMPL_MATH_UNARY_FUNCTION(exp, FLOAT) IMPL_MATH_UNARY_FUNCTION(log2, FLOAT)

namespace scaluq::internal {
CALL_MACRO_FOR_FLOAT(IMPL_ALL_MATH_UNARY_FUNCTION)

template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> sin(const Complex<Fp>& x) {
    return Complex<Fp>(sin(x.real()) * cosh(x.imag()), cos(x.real()) * sinh(x.imag()));
}

template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> cos(const Complex<Fp>& x) {
    return Complex<Fp>(cos(x.real()) * cosh(x.imag()), -sin(x.real()) * sinh(x.imag()));
}
template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> exp(const Complex<Fp>& x) {
    return exp(x.real()) * Complex<Fp>(cos(x.imag()), sin(x.imag()));
}

template <std::floating_point Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> polar(const Fp& r, const Fp& theta = Fp()) {
    assert(r >= 0);
    return Complex<Fp>(r * cos(theta), r * sin(theta));
}

}  // namespace scaluq::internal
