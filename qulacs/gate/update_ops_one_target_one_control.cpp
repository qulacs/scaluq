#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
void cnot_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT target_mask = 1ULL << target_qubit_index;
    const UINT control_mask = 1ULL << control_qubit_index;
    const UINT min_qubit_index = std::min(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = std::max(control_qubit_index, target_qubit_index);
    const UINT min_qubit_mask = 1ULL << min_qubit_index;
    const UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const UINT low_mask = min_qubit_mask - 1;
    const UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const UINT high_mask = ~(max_qubit_mask - 1);

    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & low_mask) | ((it & mid_mask) << 1) | 
            ((it & high_mask) << 2) | control_mask;
            UINT j = i | target_mask;
            Kokkos::Experimental::swap(amplitudes[i], amplitudes[j]);
        });
}

void cz_gate(UINT control_qubit_index, UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT target_mask = 1ULL << target_qubit_index;
    const UINT control_mask = 1ULL << control_qubit_index;
    const UINT min_qubit_index = std::min(control_qubit_index, target_qubit_index);
    const UINT max_qubit_index = std::max(control_qubit_index, target_qubit_index);
    const UINT min_qubit_mask = 1ULL << min_qubit_index;
    const UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    const UINT low_mask = min_qubit_mask - 1;
    const UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    const UINT high_mask = ~(max_qubit_mask - 1);

    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & low_mask) | ((it & mid_mask) << 1) | 
            ((it & high_mask) << 2) | control_mask | target_mask;
            amplitudes[i] *= -1;
        });
}
}  // namespace qulacs
