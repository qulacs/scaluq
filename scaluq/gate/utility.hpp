#pragma once

#include "gate.hpp"

namespace scaluq {
std::pair<Gate, double> merge_gate(const Gate& gate1, const Gate& gate2);
}
