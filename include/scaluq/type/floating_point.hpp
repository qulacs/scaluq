#pragma once

#include <type_traits>

#ifdef SCALUQ_USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#else
#include <stdfloat>
#endif
#include <limits>

namespace scaluq {
enum class Precision { F16, F32, F64, BF16 };
}  // namespace scaluq

namespace scaluq::internal {
template <Precision precision>
struct FloatTypeImpl {};

template <typename T>
struct IsFloatingPoint : public std::false_type {};
#ifdef SCALUQ_FLOAT16
#if defined(SCALUQ_USE_CUDA)
using F16 = __half;
#elif defined(__STDCPP_FLOAT16_T__)
using F16 = std::float16_t;
#else
static_assert(std::numeric_limits<_Float16>::is_iec559 && sizeof(_Float16) == 2,
              "Standard double precision float (IEEE 754) is required.");
using F16 = _Float16;
#endif

template <>
struct IsFloatingPoint<F16> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F16> {
    using Type = F16;
};
#endif

#ifdef SCALUQ_FLOAT32
#if defined(SCALUQ_USE_CUDA)
using F32 = float;
#elif defined(__STDCPP_FLOAT32_T__)
using F32 = std::float32_t;
#else
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "Standard double precision float (IEEE 754) is required.");
using F32 = float;
#endif

template <>
struct IsFloatingPoint<F32> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F32> {
    using Type = F32;
};
#endif

#ifdef SCALUQ_FLOAT64
#if defined(SCALUQ_USE_CUDA)
using F64 = double;
#elif defined(__STDCPP_FLOAT64_T__)
using F64 = std::float64_t;
#else
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "Standard double precision float (IEEE 754) is required.");
using F64 = double;
#endif

template <>
struct IsFloatingPoint<F64> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F64> {
    using Type = F64;
};
#endif

#ifdef SCALUQ_BFLOAT16
#if defined(SCALUQ_USE_CUDA)
using BF16 = __nv_bfloat16;
#elif defined(__STDCPP_BFLOAT16_T__)
using BF16 = std::bfloat16_t;
#else
static_assert(sizeof(__bf16) == 2, "__bf16 must be 2 bytes");
static_assert(std::numeric_limits<__bf16>::is_specialized,
              "__bf16 must have numeric_limits specialization");
static_assert(std::numeric_limits<__bf16>::max_exponent == std::numeric_limits<float>::max_exponent,
              "__bf16 must have the same exponent range as float32");
static_assert(std::numeric_limits<__bf16>::digits == 8,
              "__bf16 must have 8 bits of mantissa precision (including implicit bit)");
using BF16 = __bf16;
#endif

template <>
struct IsFloatingPoint<BF16> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::BF16> {
    using Type = BF16;
};
#endif

template <typename T>
constexpr bool IsFloatingPointV = IsFloatingPoint<T>::value;
template <typename T>
concept FloatingPoint = IsFloatingPointV<T>;
template <Precision Prec>
using Float = FloatTypeImpl<Prec>::Type;
};  // namespace scaluq::internal
