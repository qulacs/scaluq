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
    assert(upper_index > (lower_index + block_size - 1));
    assert(n_qubits > (upper_index + block_size - 1));
    const UINT kblk_dim = 1ULL << (n_qubits - upper_index);
    const UINT jblk_dim = 1ULL << (upper_index - lower_index);
    const UINT iblk_dim = 1ULL << lower_index;
    const UINT mask_block = (1 << block_size) - 1;
    auto amplitudes = state.amplitudes_raw();

    Kokkos::parallel_for(
        kblk_dim, KOKKOS_LAMBDA(const UINT& kblk) {
            const UINT kblk_masked = kblk & mask_block;
            const UINT kblk_head = kblk - kblk_masked;
            const UINT jblk_start = kblk_masked + 1;

            Kokkos::parallel_for(
                jblk_dim, KOKKOS_LAMBDA(const UINT& jblk) {
                    const UINT jblk_masked = jblk & mask_block;
                    const UINT jblk_head = jblk - jblk_masked;
                    if (jblk_masked < jblk_start) return;

                    UINT si = (kblk << upper_index) + (jblk << lower_index);
                    UINT ti = ((kblk_head + jblk_masked) << upper_index) +
                              ((jblk_head + kblk_masked) << lower_index);
                    Kokkos::parallel_for(
                        iblk_dim, KOKKOS_LAMBDA(const UINT& i) {
                            Kokkos::Experimental::swap(amplitudes[si], amplitudes[ti]);
                        });
                });
        });
}
}  // namespace qulacs
