#pragma once

#include <type_traits>
#ifndef SCALUQ_USE_CUDA
#include <iostream>
#include <stdfloat>
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
#ifdef SCALUQ_USE_CUDA
#include <cuda_fp16.h>
using F16 = __half;
#else
#ifndef __STDCPP_FLOAT16_T__
static_assert(false && "float16 is not supported")
#endif
    using F16 = std::float16_t;
#endif
template <>
struct IsFloatingPoint<F16> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F16> {
    using Type = F16;
};
#endif
#ifdef SCALUQ_FLOAT32
#ifdef SCALUQ_USE_CUDA
using F32 = float;
#else
#ifndef __STDCPP_FLOAT32_T__
static_assert(false && "float32 is not supported")
#endif
    using F32 = std::float32_t;
#endif
template <>
struct IsFloatingPoint<F32> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F32> {
    using Type = F32;
};
#endif
#ifdef SCALUQ_FLOAT64
#ifdef SCALUQ_USE_CUDA
using F64 = double;
#else
#ifndef __STDCPP_FLOAT64_T__
static_assert(false && "float64 is not supported")
#endif
    using F64 = std::float64_t;
#endif
template <>
struct IsFloatingPoint<F64> : public std::true_type {};
template <>
struct FloatTypeImpl<Precision::F64> {
    using Type = F64;
};
#endif
#ifdef SCALUQ_BFLOAT16
#ifdef SCALUQ_USE_CUDA
using BF16 = __nv_bfloat16;
#else
#ifndef __STDCPP_BFLOAT16_T__
static_assert(false && "bfloat16 is not supported")
#endif
    using BF16 = std::bfloat16_t;
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
