#pragma once

#include "gate.hpp"

namespace scaluq {
template <Precision Prec>
std::pair<Gate<Prec>, double> merge_gate(const Gate<Prec>& gate1, const Gate<Prec>& gate2);

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_merge_gate_hpp(nb::module_& m) {
    m.def("merge_gate",
          &merge_gate<Prec>,
          "Merge two gates. return value is (merged gate, global phase).");
}
}  // namespace internal
#endif
}  // namespace scaluq
