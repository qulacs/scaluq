#pragma once

#include "type/floating_point.hpp"
#include "types.hpp"

namespace scaluq::internal {
#ifndef SCALUQ_INTERNAL_PREC
constexpr Precision Prec = Precision::F64;  // dummy for code completion
static_assert("MACRO SCALUQ_INTERNAL_PREC must be defined");
#else
constexpr Precision Prec = Precision::SCALUQ_INTERNAL_PREC;
#endif

#ifndef SCALUQ_INTERNAL_SPACE
constexpr ExecutionSpace Space = ExecutionSpace::Default;  // dummy for code completion
static_assert("MACRO SCALUQ_INTERNAL_SPACE must be defined");
#else
constexpr ExecutionSpace Space = ExecutionSpace::SCALUQ_INTERNAL_SPACE;
#endif
}  // namespace scaluq::internal
