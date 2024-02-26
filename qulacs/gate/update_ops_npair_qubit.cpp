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
    auto [lower_index, upper_index] = Kokkos::minmax(target_qubit_index_0, target_qubit_index_1);
    if (n_qubits <= (upper_index + block_size - 1)) {
        throw std::runtime_error(
            "FusedSwap: num of qubits must be bigger than upper_index + block_size - 1");
    }
    const UINT mask_block = (1 << block_size) - 1;
    const UINT kblk_mask = mask_block << upper_index;
    const UINT jblk_mask = mask_block << lower_index;
    const UINT else_mask = (1 << n_qubits) - 1 - kblk_mask - jblk_mask;

    Kokkos::parallel_for(
        state.dim(), KOKKOS_LAMBDA(const UINT& i) {
            const UINT kblk = (i & kblk_mask) >> upper_index;
            const UINT jblk = (i & jblk_mask) >> lower_index;
            if (jblk > kblk) {
                const UINT j = (i & else_mask) | jblk << upper_index | kblk << lower_index;
                Kokkos::Experimental::swap(state._raw[i], state._raw[j]);
            }
        });
}
}  // namespace qulacs
