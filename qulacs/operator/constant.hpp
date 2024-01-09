#pragma once

#include "../types.hpp"

namespace qulacs {
struct {
    Complex val[4];
} PHASE_90ROT = {1., Complex(0., 1.), -1., Complex(0., -1.)};
}  // namespace qulacs
