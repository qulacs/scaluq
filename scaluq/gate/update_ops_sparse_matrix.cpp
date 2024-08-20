#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "../types.hpp"
#include "../util/utility.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "update_ops.hpp"

namespace scaluq {
namespace internal {
void sparse_matrix_gate(std::vector<UINT> target_index_list,
                        std::vector<UINT> control_index_list,
                        const CrsMatrix& matrix,
                        StateVector& state_vector) {
    if (control_index_list.size() > 0) {
        throw std::logic_error(
            "Error: SparseMatrixGateImpl::update_quantum_State(): Control qubit is not "
            "supported.");
    }

    const UINT target_qubit_index_count = target_index_list.size();
    const UINT matrix_dim = 1ULL << target_qubit_index_count;
    const UINT loop_dim = state_vector.dim() >> target_qubit_index_count;

    const std::vector<UINT> matrix_mask_list = create_matrix_mask_list(target_index_list);
    const std::vector<UINT> sorted_insert_index_list = create_sorted_ui_list(target_index_list);
    Kokkos::View<UINT*> matrix_mask_view = convert_host_vector_to_device_view(matrix_mask_list);
    Kokkos::View<UINT*> sorted_insert_index_view =
        convert_host_vector_to_device_view(sorted_insert_index_list);

    auto get_matrix_element = [=](const CrsMatrix& mat, UINT row, UINT col) {
        Complex result = 0;

        auto row_map = matrix.graph.row_map;
        auto entries = matrix.graph.entries;
        auto values = matrix.values;

        const int row_start = row_map(row);
        const int row_end = row_map(row + 1);

        int ok = row_end;
        int ng = row_start - 1;
        int mid;
        while (ok - ng > 1) {
            mid = (ok + ng) / 2;
            if (entries[mid] >= col)
                ok = mid;
            else
                ng = mid;
        }
        if ((UINT)entries[ok] == col) result = values[ok];
        return result;
    };
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(loop_dim, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            auto buffer = Kokkos::View<Complex*>("buffer", matrix_dim);
            UINT basis_0 = team.league_rank();
            for (UINT cursor = 0; cursor < target_qubit_index_count; ++cursor) {
                UINT insert_index = sorted_insert_index_view[cursor];
                basis_0 = insert_zero_to_basis_index(basis_0, insert_index);
            }

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [&](UINT y) {
                Complex sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, matrix_dim),
                    [&](UINT x, Complex& inner_sum) {
                        inner_sum += get_matrix_element(matrix, y, x) *
                                     state_vector._raw[basis_0 ^ matrix_mask_view(x)];
                    },
                    sum);
                buffer[y] = sum;
            });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, matrix_dim), [=](const UINT y) {
                state_vector._raw[basis_0 ^ matrix_mask_view(y)] = buffer[y];
            });
        });
    Kokkos::fence();
}
}  // namespace internal
}  // namespace scaluq
