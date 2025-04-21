#pragma once

#include <scaluq/types.hpp>

namespace scaluq::internal {
#ifndef SCALUQ_INTERNAL_PREC
static_assert("MACRO SCALUQ_INTERNAL_PREC must be defined");
#endif
constexpr Precision Prec = Precision::SCALUQ_INTERNAL_PREC;

#ifndef SCALUQ_INTERNAL_SPACE
static_assert("MACRO SCALUQ_INTERNAL_SPACE must be defined");
#endif
constexpr ExecutionSpace Space = ExecutionSpace::SCALUQ_INTERNAL_SPACE;
}  // namespace scaluq::internal
