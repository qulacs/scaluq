#pragma once
#include <scaluq/types.hpp>

#ifdef SCALUQ_FLOAT16
#define SCALUQ_CALL_MACRO_FLOAT16(MACRO) MACRO(::scaluq::Precision::F16)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT16(MACRO) \
    MACRO(::scaluq::internal::F16) MACRO(::scaluq::internal::Complex<::scaluq::Precision::F16>)
#define SCALUQ_DECLARE_CLASS_FLOAT16(Class) template class Class<::scaluq::Precision::F16>;
#else
#define SCALUQ_CALL_MACRO_FLOAT16(MACRO)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT16(MACRO)
#define SCALUQ_DECLARE_CLASS_FLOAT16(Class)
#endif
#ifdef SCALUQ_FLOAT32
#define SCALUQ_CALL_MACRO_FLOAT32(MACRO) MACRO(::scaluq::Precision::F32)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT32(MACRO) \
    MACRO(::scaluq::internal::F32) MACRO(::scaluq::internal::Complex<::scaluq::Precision::F32>)
#define SCALUQ_DECLARE_CLASS_FLOAT32(Class) template class Class<::scaluq::Precision::F32>;
#else
#define SCALUQ_CALL_MACRO_FLOAT32(MACRO)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT32(MACRO)
#define SCALUQ_DECLARE_CLASS_FLOAT32(Class)
#endif
#ifdef SCALUQ_FLOAT64
#define SCALUQ_CALL_MACRO_FLOAT64(MACRO) MACRO(::scaluq::Precision::F64)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT64(MACRO) \
    MACRO(::scaluq::internal::F64) MACRO(::scaluq::internal::Complex<::scaluq::Precision::F64>)
#define SCALUQ_DECLARE_CLASS_FLOAT64(Class) template class Class<::scaluq::Precision::F64>;
#else
#define SCALUQ_CALL_MACRO_FLOAT64(MACRO)
#define SCALUQ_CALL_MACRO_TYPES_FLOAT64(MACRO)
#define SCALUQ_DECLARE_CLASS_FLOAT64(Class)
#endif
#ifdef SCALUQ_BFLOAT16
#define SCALUQ_CALL_MACRO_BFLOAT16(MACRO) MACRO(::scaluq::Precision::BF16)
#define SCALUQ_CALL_MACRO_TYPES_BFLOAT16(MACRO) \
    MACRO(::scaluq::internal::BF16) MACRO(::scaluq::internal::Complex<::scaluq::Precision::BF16>)
#define SCALUQ_DECLARE_CLASS_BFLOAT16(Class) template class Class<::scaluq::Precision::BF16>;
#else
#define SCALUQ_CALL_MACRO_BFLOAT16(MACRO)
#define SCALUQ_CALL_MACRO_TYPES_BFLOAT16(MACRO)
#define SCALUQ_DECLARE_CLASS_BFLOAT16(Class)
#endif

#define SCALUQ_CALL_MACRO_FOR_PRECISION(MACRO) \
    SCALUQ_CALL_MACRO_FLOAT16(MACRO)           \
    SCALUQ_CALL_MACRO_FLOAT32(MACRO)           \
    SCALUQ_CALL_MACRO_FLOAT64(MACRO)           \
    SCALUQ_CALL_MACRO_BFLOAT16(MACRO)
#define SCALUQ_DECLARE_CLASS_FOR_PRECISION(Class) \
    SCALUQ_DECLARE_CLASS_FLOAT16(Class)           \
    SCALUQ_DECLARE_CLASS_FLOAT32(Class)           \
    SCALUQ_DECLARE_CLASS_FLOAT64(Class)           \
    SCALUQ_DECLARE_CLASS_BFLOAT16(Class)
#define SCALUQ_CALL_MACRO_FOR_TYPES(MACRO)  \
    SCALUQ_CALL_MACRO_TYPES_FLOAT16(MACRO)  \
    SCALUQ_CALL_MACRO_TYPES_FLOAT32(MACRO)  \
    SCALUQ_CALL_MACRO_TYPES_FLOAT64(MACRO)  \
    SCALUQ_CALL_MACRO_TYPES_BFLOAT16(MACRO) \
    MACRO(std::uint8_t)                     \
    MACRO(std::uint16_t)                    \
    MACRO(std::uint32_t)                    \
    MACRO(std::uint64_t)                    \
    MACRO(double)                           \
    MACRO(::scaluq::StdComplex)
