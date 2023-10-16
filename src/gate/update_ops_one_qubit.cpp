#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "update_ops.hpp"

void x_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
    Kokkos::parallel_for(
        1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const UINT& it) {
            UINT i = (it & high_mask) << 1 | (it & low_mask);
            Kokkos::Experimental::swap(state[i], state[i | (1ULL << target_qubit_index)]);
        });
}

void y_gate(UINT target_qubit_index, StateVector& state) {}

void z_gate(UINT target_qubit_index, StateVector& state) {
    const UINT n_qubits = state.n_qubits();
    const UINT low_mask = (1ULL << target_qubit_index) - 1;
    const UINT high_mask = ~low_mask;
}
