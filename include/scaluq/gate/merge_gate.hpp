#pragma once

#include "gate.hpp"

namespace scaluq {
template <std::floating_point Fp, ExecutionSpace Sp>
std::pair<Gate<Fp, Sp>, Fp> merge_gate(const Gate<Fp, Sp>& gate1, const Gate<Fp, Sp>& gate2);

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_merge_gate_hpp(nb::module_& m) {
    m.def("merge_gate",
          &merge_gate<double>,
          "Merge two gates. return value is (merged gate, global phase).");
}
}  // namespace internal
#endif
}  // namespace scaluq
