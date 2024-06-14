#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"
#include "util/utility.hpp"

namespace scaluq {
namespace internal {
void cx_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 2, KOKKOS_LAMBDA(UINT it) {
            UINT i =
                internal::insert_zero_to_basis_index(it, target_qubit_index, control_qubit_index);
            i |= 1ULL << control_qubit_index;
            Kokkos::Experimental::swap(state._raw[i], state._raw[i | (1ULL << target_qubit_index)]);
        });
    Kokkos::fence();
}

void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 2, KOKKOS_LAMBDA(UINT it) {
            UINT i =
                internal::insert_zero_to_basis_index(it, target_qubit_index, control_qubit_index);
            i |= 1ULL << control_qubit_index;
            i |= 1ULL << target_qubit_index;
            state._raw[i] *= -1;
        });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
