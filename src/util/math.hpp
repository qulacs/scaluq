#pragma once

#include <cmath>
#include <scaluq/types.hpp>

#include "../prec_space.hpp"

#ifndef SCALUQ_USE_CUDA
#define IMPL_MATH_UNARY_FUNCTION(FUNC) \
    KOKKOS_INLINE_FUNCTION Float<Prec> FUNC(Float<Prec> x) { return std::FUNC(x); }

namespace scaluq::internal {
IMPL_MATH_UNARY_FUNCTION(sqrt)
IMPL_MATH_UNARY_FUNCTION(sin)
IMPL_MATH_UNARY_FUNCTION(cos)
IMPL_MATH_UNARY_FUNCTION(exp)
IMPL_MATH_UNARY_FUNCTION(log2)
IMPL_MATH_UNARY_FUNCTION(sinh)
IMPL_MATH_UNARY_FUNCTION(cosh)
}  // namespace scaluq::internal

#undef IMPL_ALL_MATH_UNARY_FUNCTION
#undef IMPL_MATH_UNARY_FUNCTION
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

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Complex<Prec> sin(const Complex<Prec>& x) {
    return Complex<Prec>(sin(x.real()) * cosh(x.imag()), cos(x.real()) * sinh(x.imag()));
}

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Complex<Prec> cos(const Complex<Prec>& x) {
    return Complex<Prec>(cos(x.real()) * cosh(x.imag()), -sin(x.real()) * sinh(x.imag()));
}

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Complex<Prec> exp(const Complex<Prec>& x) {
    return exp(x.real()) * Complex<Prec>(cos(x.imag()), sin(x.imag()));
}

template <Precision Prec>
KOKKOS_INLINE_FUNCTION Complex<Prec> polar(const Float<Prec>& r,
                                           const Float<Prec>& theta = Float<Prec>()) {
    assert(r >= Float<Prec>{0});
    return Complex<Prec>(r * cos(theta), r * sin(theta));
}

template <class Scalar, ExecutionSpace Space>
struct Sum {
public:
    // Required
    using reducer = Sum<Scalar, Space>;
    using value_type = std::remove_cv_t<Scalar>;
    static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

    using result_view_type = Kokkos::View<value_type, SpaceType<Space>>;

private:
    result_view_type value;
    bool references_scalar_v;

public:
    KOKKOS_INLINE_FUNCTION
    Sum(value_type& value_) : value(&value_), references_scalar_v(true) {}

    KOKKOS_INLINE_FUNCTION
    Sum(const result_view_type& value_) : value(value_), references_scalar_v(false) {}

    // Required
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const { dest += src; }

    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const { val = static_cast<value_type>(0); }

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const { return *value.data(); }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const { return value; }

    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const { return references_scalar_v; }
};
}  // namespace scaluq::internal
