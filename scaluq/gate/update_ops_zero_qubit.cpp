#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void i_gate(StateVector&) {}

void global_phase_gate(double phase, StateVector& state) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(const UINT& i) { state._raw[i] *= coef; });
}
}  // namespace internal
}  // namespace scaluq
