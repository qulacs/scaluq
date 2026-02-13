#pragma once

#include <limits>
#include <type_traits>

#ifdef SCALUQ_USE_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#elif __has_include(<stdfloat>)
#include <stdfloat>
#define STDFLOAT_ENABLED 1
#else
#define STDFLOAT_ENABLED 0
#endif

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
#elif STDFLOAT_ENABLED && defined(__STDCPP_FLOAT16_T__)
using F16 = std::float16_t;
#else
static_assert(false, "This compiler does not support standard F16 type.");
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
#elif STDFLOAT_ENABLED && defined(__STDCPP_FLOAT32_T__)
using F32 = std::float32_t;
#else
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
              "standard single precision float (IEEE 754) is required");
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
#elif STDFLOAT_ENABLED && defined(__STDCPP_FLOAT64_T__)
using F64 = std::float64_t;
#else
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
              "standard double precision float (IEEE 754) is required");
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
#elif STDFLOAT_ENABLED && defined(__STDCPP_BFLOAT16_T__)
using BF16 = std::bfloat16_t;
#else
static_assert(false, "This compiler does not support standard BF16 type.");
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
