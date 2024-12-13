#pragma once
#include <scaluq/types.hpp>

#define FLOAT(Fp) template <::scaluq::FloatingPoint Fp>

#ifdef SCALUQ_FLOAT16
#define CALL_MACRO_FLOAT16(MACRO) MACRO(F16)
#define CALL_MACRO_CFLOAT16(MACRO) MACRO(scaluq::Complex<scaluq::F16>)
#define DECLARE_CLASS_FLOAT16(Class) template class Class<scaluq::F16>;
#else
#define CALL_MACRO_FLOAT16(MACRO)
#define CALL_MACRO_CFLOAT16(MACRO)
#define DECLARE_CLASS_FLOAT16(Class)
#endif
#ifdef SCALUQ_FLOAT32
#define CALL_MACRO_FLOAT32(MACRO) MACRO(F32)
#define CALL_MACRO_CFLOAT32(MACRO) MACRO(scaluq::Complex<scaluq::F32>)
#define DECLARE_CLASS_FLOAT32(Class) template class Class<scaluq::F32>;
#else
#define CALL_MACRO_FLOAT32(MACRO)
#define CALL_MACRO_CFLOAT32(MACRO)
#define DECLARE_CLASS_FLOAT32(Class)
#endif
#ifdef SCALUQ_FLOAT64
#define CALL_MACRO_FLOAT64(MACRO) MACRO(F64)
#define CALL_MACRO_CFLOAT64(MACRO) MACRO(scaluq::Complex<scaluq::F64>)
#define DECLARE_CLASS_FLOAT64(Class) template class Class<scaluq::F64>;
#else
#define CALL_MACRO_FLOAT64(MACRO)
#define CALL_MACRO_CFLOAT64(MACRO)
#define DECLARE_CLASS_FLOAT64(Class)
#endif
#ifdef SCALUQ_BFLOAT16
#define CALL_MACRO_BFLOAT16(MACRO) MACRO(BF16)
#define CALL_MACRO_CBFLOAT16(MACRO) MACRO(scaluq::Complex<scaluq::BF16>)
#define DECLARE_CLASS_BFLOAT16(Class) template class Class<scaluq::BF16>;
#else
#define CALL_MACRO_BFLOAT16(MACRO)
#define CALL_MACRO_CBFLOAT16(MACRO)
#define DECLARE_CLASS_BFLOAT16(Class)
#endif

#define CALL_MACRO_FOR_FLOAT(MACRO) \
    CALL_MACRO_FLOAT16(MACRO)       \
    CALL_MACRO_FLOAT32(MACRO)       \
    CALL_MACRO_FLOAT64(MACRO)       \
    CALL_MACRO_BFLOAT16(MACRO)
#define CALL_MACRO_FOR_COMPLEX(MACRO) \
    CALL_MACRO_CFLOAT16(MACRO)        \
    CALL_MACRO_CFLOAT32(MACRO)        \
    CALL_MACRO_CFLOAT64(MACRO)        \
    CALL_MACRO_CBFLOAT16(MACRO)
#define CALL_MACRO_FOR_UINT(MACRO) \
    MACRO(std::uint8_t) MACRO(std::uint16_t) MACRO(std::uint32_t) MACRO(std::uint64_t)
#define FLOAT_DECLARE_CLASS(Class) \
    DECLARE_CLASS_FLOAT16(Class)   \
    DECLARE_CLASS_FLOAT32(Class)   \
    DECLARE_CLASS_FLOAT64(Class)   \
    DECLARE_CLASS_BFLOAT16(Class)

namespace scaluq::internal {
template <class Scalar, class Space>
struct Sum {
public:
    // Required
    using reducer = Sum<Scalar, Space>;
    using value_type = std::remove_cv_t<Scalar>;
    static_assert(!std::is_pointer_v<value_type> && !std::is_array_v<value_type>);

    using result_view_type = Kokkos::View<value_type, Space>;

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
