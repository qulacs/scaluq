#pragma once

#include <type_traits>
#ifndef SCALUQ_USE_CUDA
#include <iostream>
#include <stdfloat>
#endif

namespace scaluq {
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
#endif
template <typename T>
constexpr bool IsFloatingPointV = IsFloatingPoint<T>::value;
template <typename T>
concept FloatingPoint = IsFloatingPointV<T>;
};  // namespace scaluq

#ifdef SCALUQ_USE_CUDA
#ifdef SCALUQ_FLOAT16
inline std::ostream& operator<<(std::ostream& out, scaluq::F16 x) { return out << __half2float(x); }
namespace nlohmann {
template <>
struct adl_serializer<::scaluq::F16> {
    static void to_json(json& j, const ::scaluq::F16& x) { j = __half2float(x); }
    static void from_json(const json& j, scaluq::F16& x) {
        float f;
        j.get_to(f);
        x = __float2half(f);
    }
};
}  // namespace nlohmann
#endif
#ifdef SCALUQ_BFLOAT16
inline std::ostream& operator<<(std::ostream& out, scaluq::BF16 x) {
    return out << __bfloat162float(x);
}
namespace nlohmann {
template <>
struct adl_serializer<::scaluq::BF16> {
    static void to_json(json& j, const ::scaluq::BF16& x) { j = __bfloat162float(x); }
    static void from_json(const json& j, scaluq::BF16& x) {
        float f;
        j.get_to(f);
        x = __float2bfloat16(f);
    }
};
}  // namespace nlohmann
#endif
#endif
