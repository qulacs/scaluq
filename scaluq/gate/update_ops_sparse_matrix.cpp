#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"

namespace scaluq {
namespace internal {
void sparse_matrix_gate(std::vector<UINT> target_index_list,
                        std::vector<UINT> control_index_list,
                        std::vector<UINT> control_value_list,
                        const CrsMatrix& matrix,
                        StateVector& state_vector) {
    if (control_index_list.size() > 0) {
        throw std::logic_error(
            "Error: SparseMatrixGateImpl::update_quantum_State(): Control qubit is not "
            "supported.");
    }
    Kokkos::View<scaluq::Complex*> state_view = state_vector._raw;

    const UINT target_qubit_index_count = target_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    const UINT loop_dim = state_vector.dim() >> target_qubit_index_count;

    const std::vector<UINT> matrix_mask_list = create_matrix_mask_list(target_index_list);
    const std::vector<UINT> sorted_insert_index_list = create_sorted_ui_list(target_index_list);
    Kokkos::View<UINT*> matrix_mask_view = convert_host_vector_to_device_view(matrix_mask_list);

    Kokkos::View<scaluq::Complex*> buffer1("buffer1", matrix_dim);
    Kokkos::View<scaluq::Complex*> buffer2("buffer2", matrix_dim);

    for (UINT state_index = 0; state_index < loop_dim; ++state_index) {
        UINT basis_0 = state_index;
        // create base index
        for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
            UINT insert_index = sorted_insert_index_list[cursor];
            basis_0 = insert_zero_to_basis_index(basis_0, insert_index);
        }

        // fetch vector
        Kokkos::parallel_for(
            matrix_dim,
            KOKKOS_LAMBDA(UINT j) { buffer1[j] = state_view[basis_0 ^ matrix_mask_view(j)]; });
        Kokkos::fence();

        spmv(matrix, buffer1, buffer2);

        // set result
        Kokkos::parallel_for(
            matrix_dim,
            KOKKOS_LAMBDA(UINT j) { state_view[basis_0 ^ matrix_mask_view(j)] = buffer2[j]; });
        Kokkos::fence();
    }
}
}  // namespace internal
}  // namespace scaluq
