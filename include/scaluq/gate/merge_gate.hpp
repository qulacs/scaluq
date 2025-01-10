#pragma once

#include "gate.hpp"

namespace scaluq {
template <FloatingPoint Fp>
inline std::pair<Gate<Fp>, Fp> merge_gate(const Gate<Fp>& gate1, const Gate<Fp>& gate2);

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <FloatingPoint Fp>
void bind_gate_merge_gate_hpp(nb::module_& m) {
    m.def("merge_gate",
          &merge_gate<Fp>,
          "Merge two gates. return value is (merged gate, global phase).");
}
}  // namespace internal
#endif
}  // namespace scaluq
