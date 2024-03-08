#include "update_ops.hpp"

namespace qulacs {
namespace internal {
void i_gate(StateVector&) {}
void global_phase_gate(double angle, StateVector& state) {
    Complex phase = Kokkos::polar(1., angle);
    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(const UINT& i) { state._raw[i] *= phase; });
}
}  // namespace internal
}  // namespace qulacs
