#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void i_gate(UINT, UINT, StateVector&) {}

void global_phase_gate(UINT, UINT control_mask, double phase, StateVector& state) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        state.dim() >> std::popcount(control_mask), KOKKOS_LAMBDA(UINT i) {
            state._raw[insert_zero_at_mask_positions(i, control_mask) | control_mask] *= coef;
        });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
