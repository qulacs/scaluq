#pragma once

#include "gate.hpp"

namespace scaluq {
<<<<<<< HEAD
template <Precision Prec>
inline std::pair<Gate<Prec>, double> merge_gate(const Gate<Prec>& gate1, const Gate<Prec>& gate2);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
std::pair<Gate<Fp, Sp>, Fp> merge_gate(const Gate<Fp, Sp>& gate1, const Gate<Fp, Sp>& gate2);
>>>>>>> set-space

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
