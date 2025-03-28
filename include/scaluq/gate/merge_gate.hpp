#pragma once

#include "gate.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
std::pair<Gate<Prec, Space>, double> merge_gate(const Gate<Prec, Space>& gate1,
                                                const Gate<Prec, Space>& gate2);

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_merge_gate_hpp(nb::module_& m) {
    m.def("merge_gate",
          &merge_gate<Prec, Space>,
          "Merge two gates. return value is (merged gate, global phase).");
}
}  // namespace internal
#endif
}  // namespace scaluq
