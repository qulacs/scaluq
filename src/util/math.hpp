#pragma once

#include <cmath>
#include <scaluq/types.hpp>

#include "template.hpp"

#ifndef SCALUQ_USE_CUDA
#define IMPL_MATH_UNARY_FUNCTION(FUNC, FLOAT) \
    KOKKOS_INLINE_FUNCTION FLOAT FUNC(FLOAT x) { return std::FUNC(x); }

#define IMPL_ALL_MATH_UNARY_FUNCTION(FLOAT) \
    IMPL_MATH_UNARY_FUNCTION(sqrt, FLOAT)   \
    IMPL_MATH_UNARY_FUNCTION(sin, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(cos, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(exp, FLOAT)    \
    IMPL_MATH_UNARY_FUNCTION(log2, FLOAT)

namespace scaluq::internal {
CALL_MACRO_FOR_FLOAT(IMPL_ALL_MATH_UNARY_FUNCTION)

#undef IMPL_ALL_MATH_UNARY_FUNCTION
#undef IMPL_MATH_UNARY_FUNCTION
}  // namespace scaluq::internal
#else
#define DEFINE_NORMAL(FUNC, FLOAT) \
    KOKKOS_INLINE_FUNCTION FLOAT FUNC(FLOAT x) { return std::FUNC(x); }
#ifdef __CUDA_ARCH__
#define DEFINE_F16(FUNC) \
    KOKKOS_INLINE_FUNCTION F16 FUNC(F16 x) { return h##FUNC(x); }
#else
#define DEFINE_F16(FUNC) \
    KOKKOS_INLINE_FUNCTION F16 FUNC(F16 x) { return __float2half(std::FUNC(__half2float(x))); }
#endif
#ifdef __CUDA_ARCH__
#define DEFINE_BF16(FUNC) \
    KOKKOS_INLINE_FUNCTION BF16 FUNC(BF16 x) { return h##FUNC(x); }
#else
#define DEFINE_BF16(FUNC)                                        \
    KOKKOS_INLINE_FUNCTION BF16 FUNC(BF16 x) {                   \
        return __float2bfloat16(std::FUNC(__bfloat162float(x))); \
    }
#endif
#define DEFINE_HYPERBOLIC(FLOAT)                                                         \
    KOKKOS_INLINE_FUNCTION FLOAT sinh(FLOAT x) { return (exp(x) - exp(-x)) / FLOAT{2}; } \
    KOKKOS_INLINE_FUNCTION FLOAT cosh(FLOAT x) { return (exp(x) + exp(-x)) / FLOAT{2}; }
namespace scaluq::internal {
#ifdef SCALUQ_FLOAT16
DEFINE_F16(sqrt)
DEFINE_F16(sin)
DEFINE_F16(cos)
DEFINE_F16(exp)
DEFINE_F16(log2)
DEFINE_HYPERBOLIC(F16)
#endif
#ifdef SCALUQ_FLOAT32
DEFINE_NORMAL(sqrt, F32)
DEFINE_NORMAL(sin, F32)
DEFINE_NORMAL(cos, F32)
DEFINE_NORMAL(exp, F32)
DEFINE_NORMAL(log2, F32)
DEFINE_NORMAL(sinh, F32)
DEFINE_NORMAL(cosh, F32)
#endif
#ifdef SCALUQ_FLOAT64
DEFINE_NORMAL(sqrt, F64)
DEFINE_NORMAL(sin, F64)
DEFINE_NORMAL(cos, F64)
DEFINE_NORMAL(exp, F64)
DEFINE_NORMAL(log2, F64)
DEFINE_NORMAL(sinh, F64)
DEFINE_NORMAL(cosh, F64)
#endif
#ifdef SCALUQ_BFLOAT16
DEFINE_BF16(sqrt)
DEFINE_BF16(sin)
DEFINE_BF16(cos)
DEFINE_BF16(exp)
DEFINE_BF16(log2)
DEFINE_HYPERBOLIC(BF16)
#endif

#undef DEFINE_NORMAL
#undef DEFINE_F16
#undef DEFINE_BF16
#undef DEFINE_HYPERBOLIC
}  // namespace scaluq::internal
#endif

namespace scaluq::internal {
template <FloatingPoint Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> sin(const Complex<Fp>& x) {
    return Complex<Fp>(sin(x.real()) * cosh(x.imag()), cos(x.real()) * sinh(x.imag()));
}

template <FloatingPoint Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> cos(const Complex<Fp>& x) {
    return Complex<Fp>(cos(x.real()) * cosh(x.imag()), -sin(x.real()) * sinh(x.imag()));
}
template <FloatingPoint Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> exp(const Complex<Fp>& x) {
    return exp(x.real()) * Complex<Fp>(cos(x.imag()), sin(x.imag()));
}

template <FloatingPoint Fp>
KOKKOS_INLINE_FUNCTION Complex<Fp> polar(const Fp& r, const Fp& theta = Fp()) {
    assert(r >= Fp{0});
    return Complex<Fp>(r * cos(theta), r * sin(theta));
}
}  // namespace scaluq::internal
