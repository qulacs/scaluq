#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "update_ops.hpp"

namespace qulacs {
void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    std::array<Complex, 4> matrix,
                                    StateVector& state) {
    const UINT mask = 1ULL << target_qubit_index;
    const UINT mask_low = mask - 1;
    const UINT mask_high = ~mask_low;
    auto amplitudes = state.amplitudes_raw();
    Kokkos::parallel_for(
        state.dim() - 1, KOKKOS_LAMBDA(const UINT it) {
            UINT basis_0 = (it & mask_low) + ((it & mask_high) << 1);
            UINT basis_1 = basis_0 + mask;
            Complex val0 = state[basis_0];
            Complex val1 = state[basis_1];
            Complex res0 = matrix[0] * val0 + matrix[1] * val1;
            Complex res1 = matrix[2] * val0 + matrix[3] * val1;
            amplitudes[basis_0] = res0;
            amplitudes[basis_1] = res1;
        });
}
}  // namespace qulacs
