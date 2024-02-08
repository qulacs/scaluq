#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cassert>

#include "../types.hpp"
#include "update_ops.hpp"

namespace qulacs {
void fusedswap_gate(UINT target_qubit_index_0,
                    UINT target_qubit_index_1,
                    UINT block_size,
                    StateVector& state) {
    UINT n_qubits = state.n_qubits();
    UINT upper_index, lower_index;
    upper_index = std::max(target_qubit_index_0, target_qubit_index_1);
    lower_index = std::min(target_qubit_index_0, target_qubit_index_1);
    if (n_qubits <= (upper_index + block_size - 1)) {
        throw std::runtime_error(
            "FusedSwap: num of qubits must be bigger than upper_index + block_size - 1");
    }
    const UINT mask_block = (1 << block_size) - 1;
    auto amplitudes = state.amplitudes_raw();
    const UINT kblk_mask = mask_block << upper_index;
    const UINT jblk_mask = mask_block << lower_index;
    const UINT else_mask = (1 << n_qubits) - 1 - kblk_mask - jblk_mask;

    Kokkos::parallel_for(
        1 << n_qubits, KOKKOS_LAMBDA(const UINT& i) {
            const UINT kblk = (i & kblk_mask) >> upper_index;
            const UINT jblk = (i & jblk_mask) >> lower_index;
            if (jblk > kblk) {
                const UINT index = (i & else_mask) | jblk << upper_index | kblk << lower_index;
                Kokkos::Experimental::swap(amplitudes[i], amplitudes[index]);
            }
        });
}
}  // namespace qulacs
