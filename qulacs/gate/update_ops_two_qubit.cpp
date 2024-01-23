#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "constant.hpp"
#include "update_ops.hpp"

namespace qulacs {
void swap_gate(UINT target0, UINT target1, StateVector& state) {
    UINT n_qubits = state.n_qubits();
    UINT mask_0 = 1ULL << target0;
    UINT mask_1 = 1ULL << target1;
    UINT mask = mask_0 + mask_1;
    UINT min_qubit_index = std::min(target0, target1);
    UINT max_qubit_index = std::max(target0, target1);
    UINT min_qubit_mask = 1ULL << min_qubit_index;
    UINT max_qubit_mask = 1ULL << (max_qubit_index - 1);
    UINT low_mask = min_qubit_mask - 1;
    UINT mid_mask = (max_qubit_mask - 1) ^ low_mask;
    UINT high_mask = ~(max_qubit_mask - 1);
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const UINT& it) {
            UINT basis_index_0 =
                (it & low_mask) + ((it & mid_mask) << 1) + ((it & high_mask) << 2) + mask_0;
            UINT basis_index_1 = basis_index_0 ^ mask;
            Kokkos::Experimental::swap(amplitudes[basis_index_0], amplitudes[basis_index_1]);
        });
}
}  // namespace qulacs
