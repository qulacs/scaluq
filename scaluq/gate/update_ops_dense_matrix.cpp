#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "update_ops.hpp"
#include "util/utility.hpp"

namespace scaluq {
namespace internal {
void single_qubit_dense_matrix_gate(UINT target_qubit_index,
                                    const matrix_2_2& matrix,
                                    StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 1, KOKKOS_LAMBDA(const UINT it) {
            UINT basis_0 = internal::insert_zero_to_basis_index(it, target_qubit_index);
            UINT basis_1 = basis_0 | (1ULL << target_qubit_index);
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
        });
}

void double_qubit_dense_matrix_gate(UINT target0,
                                    UINT target1,
                                    const matrix_4_4& matrix,
                                    StateVector& state) {
    Kokkos::parallel_for(
        state.dim() >> 2, KOKKOS_LAMBDA(const UINT it) {
            UINT basis_0 = internal::insert_zero_to_basis_index(it, target0, target1);
            UINT basis_1 = basis_0 | (1ULL << target0);
            UINT basis_2 = basis_0 | (1ULL << target1);
            UINT basis_3 = basis_1 | (1ULL << target1);
            Complex val0 = state._raw[basis_0];
            Complex val1 = state._raw[basis_1];
            Complex val2 = state._raw[basis_2];
            Complex val3 = state._raw[basis_3];
            Complex res0 = matrix.val[0][0] * val0 + matrix.val[0][1] * val1 +
                           matrix.val[0][2] * val2 + matrix.val[0][3] * val3;
            Complex res1 = matrix.val[1][0] * val0 + matrix.val[1][1] * val1 +
                           matrix.val[1][2] * val2 + matrix.val[1][3] * val3;
            Complex res2 = matrix.val[2][0] * val0 + matrix.val[2][1] * val1 +
                           matrix.val[2][2] * val2 + matrix.val[2][3] * val3;
            Complex res3 = matrix.val[3][0] * val0 + matrix.val[3][1] * val1 +
                           matrix.val[3][2] * val2 + matrix.val[3][3] * val3;
            state._raw[basis_0] = res0;
            state._raw[basis_1] = res1;
            state._raw[basis_2] = res2;
            state._raw[basis_3] = res3;
        });
}
}  // namespace internal
}  // namespace scaluq
