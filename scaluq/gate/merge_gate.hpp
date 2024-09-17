#pragma once

#include "gate.hpp"

namespace scaluq {
std::pair<Gate, double> merge_gate(const Gate& gate1, const Gate& gate2);

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_merge_gate_hpp(nb::module_& m) {
    m.def(
        "merge_gate", &merge_gate, "Merge two gates. return value is (merged gate, global phase).");
}
}  // namespace internal
#endif
}  // namespace scaluq
