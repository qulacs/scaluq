#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../constant.hpp"
#include "../types.hpp"
#include "../util/utility.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void swap_gate(UINT target_mask, UINT control_mask, StateVector& state) {
    UINT lower_target_mask = -target_mask & target_mask;
    UINT upper_target_mask = target_mask ^ lower_target_mask;
    Kokkos::parallel_for(
        state.n_qubits() >> std::popcount(target_mask | control_mask), KOKKOS_LAMBDA(UINT it) {
            UINT basis =
                insert_zero_at_mask_positions(it, target_mask | control_mask) | control_mask;
            Kokkos::Experimental::swap(state._raw[basis | lower_target_mask],
                                       state._raw[basis | upper_target_mask]);
        });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
