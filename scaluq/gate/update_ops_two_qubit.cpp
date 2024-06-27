#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../constant.hpp"
#include "../types.hpp"
#include "update_ops.hpp"
#include "util/utility.hpp"

namespace scaluq {
namespace internal {
void swap_gate(UINT target0, UINT target1, StateVector& state) {
    Kokkos::parallel_for(
        1ULL << (state.n_qubits() - 2), KOKKOS_LAMBDA(UINT it) {
            UINT basis = internal::insert_zero_to_basis_index(it, target0, target1);
            Kokkos::Experimental::swap(state._raw[basis | (1ULL << target0)],
                                       state._raw[basis | (1ULL << target1)]);
        });
    Kokkos::fence();
}
void swap_gate(UINT target0, UINT target1, StateVectorBatched& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim() >> 2}),
        KOKKOS_LAMBDA(UINT batch_id, UINT it) {
            UINT basis = internal::insert_zero_to_basis_index(it, target0, target1);
            Kokkos::Experimental::swap(states._raw(batch_id, basis | (1ULL << target0)),
                                       states._raw(batch_id, basis | (1ULL << target1)));
        });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
