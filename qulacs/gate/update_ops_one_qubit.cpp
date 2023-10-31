#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "update_ops.hpp"

namespace qulacs {
void x_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    auto amplitudes = state.amplitudes_raw();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Kokkos::Experimental::swap(amplitudes[i], amplitudes[i | (1ULL << target_qubit_index)]);
        });
}
}  // namespace qulacs
