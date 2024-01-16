#pragma once

#include "../types.hpp"

namespace qulacs {
KOKKOS_INLINE_FUNCTION array_4 PHASE_90ROT() { return {1., Complex(0., 1.), -1., Complex(0., -1.)}; }
}  // namespace qulacs
