#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void i_gate(StateVector&) {}
void i_gate(StateVectorBatched&) {}

void global_phase_gate(double phase, StateVector& state) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(UINT i) { state._raw[i] *= coef; });
}

void global_phase_gate(double phase, StateVectorBatched& states) {
    Complex coef = Kokkos::polar(1., phase);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(UINT batch_id, UINT i) { states._raw(batch_id, i) *= coef; });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
