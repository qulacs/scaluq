#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../types.hpp"
#include "KokkosSparse_spmv.hpp"

namespace scaluq {

namespace internal {

/**
 * Inserts a 0 bit at a specified index in basis_index.
 * Example: insert_zero_to_basis_index(0b1001, 1) -> 0b10001.
 *                                                        ^
 */
KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
    UINT mask = (1ULL << insert_index) - 1;
    UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

/**
 * Inserts two 0 bits at specified indexes in basis_index.
 * Example: insert_zero_to_basis_index(0b11001, 1, 5) -> 0b1010001.
 *                                                          ^   ^
 */
KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index,
                                                       UINT insert_index1,
                                                       UINT insert_index2) {
    auto [lidx, uidx] = Kokkos::minmax(insert_index1, insert_index2);
    UINT lmask = (1ULL << lidx) - 1;
    UINT umask = (1ULL << uidx) - 1;
    basis_index = ((basis_index >> lidx) << (lidx + 1)) | (basis_index & lmask);
    return ((basis_index >> uidx) << (uidx + 1)) | (basis_index & umask);
}

inline std::optional<ComplexMatrix> get_pauli_matrix(PauliOperator pauli) {
    ComplexMatrix mat;
    std::vector<UINT> pauli_id_list = pauli.get_pauli_id_list();
    UINT flip_mask, phase_mask, rot90_count;
    Kokkos::parallel_reduce(
        pauli_id_list.size(),
        KOKKOS_LAMBDA(const UINT& i, UINT& f_mask, UINT& p_mask, UINT& rot90_cnt) {
            UINT pauli_id = pauli_id_list[i];
            if (pauli_id == 1) {
                f_mask ^= 1ULL << i;
            } else if (pauli_id == 2) {
                f_mask ^= 1ULL << i;
                p_mask ^= 1ULL << i;
                rot90_cnt++;
            } else if (pauli_id == 3) {
                p_mask ^= 1ULL << i;
            }
        },
        flip_mask,
        phase_mask,
        rot90_count);
    std::vector<StdComplex> rot = {1, -1.i, -1, 1.i};
    UINT matrix_dim = 1ULL << pauli_id_list.size();
    for (UINT index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1. - 2. * (Kokkos::popcount(index & phase_mask) % 2);
        mat(index, index ^ flip_mask) = rot[rot90_count % 4] * sign;
    }
    return mat;
}

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
Kokkos::View<T*, Kokkos::DefaultExecutionSpace> convert_host_vector_to_device_view(
    const std::vector<T>& vec) {
    Kokkos::fence();
    Kokkos::View<T*, Kokkos::HostSpace> host_view("host_view", vec.size());
    std::copy(vec.begin(), vec.end(), host_view.data());
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view) {
    Kokkos::fence();
    std::vector<T> host_vector(device_view.extent(0));
    Kokkos::View<T*, Kokkos::HostSpace> host_view(
        Kokkos::ViewAllocateWithoutInitializing("host_view"), device_view.extent(0));
    Kokkos::deep_copy(host_view, device_view);
    std::copy(host_view.data(), host_view.data() + host_view.extent(0), host_vector.begin());
    return host_vector;
}

}  // namespace internal

KOKKOS_INLINE_FUNCTION double squared_norm(const Complex& z) {
    return z.real() * z.real() + z.imag() * z.imag();
}

inline std::vector<UINT> create_matrix_mask_list(const std::vector<UINT> qubit_index_list,
                                                 const UINT qubit_index_count) {
    const UINT matrix_dim = 1ULL << qubit_index_count;
    std::vector<UINT> mask_list(matrix_dim, 0);

    Kokkos::parallel_for(
        matrix_dim, KOKKOS_LAMBDA(const UINT& i) {
            for (UINT j = 0; j < qubit_index_count; j++) {
                if ((i >> j) & 1) {
                    mask_list[i] ^= 1ULL << qubit_index_list[j];
                }
            }
        });
    // for (int i = 0; i < matrix_dim; i++) {
    //     for (int j = 0; j < qubit_index_count; j++) {
    //         if ((i >> j) & 1) {
    //             mask_list[i] ^= 1ULL << qubit_index_list[j];
    //         }
    //     }
    // }
    return mask_list;
}

inline std::vector<UINT> create_sorted_ui_list_value(const std::vector<UINT>& list) {
    std::vector<UINT> sorted_list(list);
    std::sort(sorted_list.begin(), sorted_list.end());
    return sorted_list;
}

// x: state vector. output will be stored in y
inline void spmv(const CrsMatrix& matrix,
                 const Kokkos::View<Complex*>& x,
                 Kokkos::View<Complex*>& y) {
    KokkosSparse::spmv("N", 1.0, matrix, x, 0.0, y);
}

// x: state vector, output will be stored in y
inline void gemv(const DenseMatrix matrix,
                 const Kokkos::View<Complex*>& x,
                 Kokkos::View<Complex*>& y) {
    KokkosBlas::gemv("N", 1.0, matrix, x, 0.0, y);
}
}  // namespace scaluq
