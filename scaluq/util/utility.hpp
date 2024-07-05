#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <ranges>
#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../types.hpp"

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

KOKKOS_INLINE_FUNCTION matrix_2_2 matrix_multiply(const matrix_2_2& matrix1,
                                                  const matrix_2_2& matrix2) {
    return {matrix1.val[0][0] * matrix2.val[0][0] + matrix1.val[0][1] * matrix2.val[1][0],
            matrix1.val[0][0] * matrix2.val[0][1] + matrix1.val[0][1] * matrix2.val[1][1],
            matrix1.val[1][0] * matrix2.val[0][0] + matrix1.val[1][1] * matrix2.val[1][0],
            matrix1.val[1][0] * matrix2.val[0][1] + matrix1.val[1][1] * matrix2.val[1][1]};
}

inline ComplexMatrix kronecker_product(const ComplexMatrix& lhs, const ComplexMatrix& rhs) {
    ComplexMatrix result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(), rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

inline ComplexMatrix get_expanded_matrix(const ComplexMatrix& from_matrix,
                                         const std::vector<UINT>& from_targets,
                                         std::vector<UINT>& to_targets) {
    /* std::cerr << "expand" << std::endl; */
    /* std::cerr << "from_targets"; */
    /* for (UINT from : from_targets) std::cerr << " " << from; */
    /* std::cerr << std::endl; */
    /* std::cerr << "to_targets"; */
    /* for (UINT to : to_targets) std::cerr << " " << to; */
    /* std::cerr << std::endl; */
    std::vector<UINT> targets_map(from_targets.size());
    std::ranges::transform(from_targets, targets_map.begin(), [&](UINT x) {
        return std::ranges::lower_bound(to_targets, x) - to_targets.begin();
    });
    /* std::cerr << "targets_map"; */
    /* for (UINT t : targets_map) std::cerr << " " << t; */
    /* std::cerr << std::endl; */
    std::vector<UINT> idx_map(1ULL << from_targets.size());
    for (UINT i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (UINT j : std::views::iota(0ULL, from_targets.size())) {
            idx_map[i] |= (i >> j & 1) << targets_map[j];
        }
    }
    /* std::cerr << "idx_map"; */
    /* for (UINT t : idx_map) std::cerr << " " << t; */
    /* std::cerr << std::endl; */

    UINT targets_idx_mask = idx_map.back();
    std::vector<UINT> outer_indices;
    outer_indices.reserve(1ULL << (to_targets.size() - from_targets.size()));
    for (UINT i : std::views::iota(0ULL, 1ULL << to_targets.size())) {
        if ((i & targets_idx_mask) == 0) outer_indices.push_back(i);
    }
    /* std::cerr << "outer_indices"; */
    /* for (UINT t : outer_indices) std::cerr << " " << t; */
    /* std::cerr << std::endl; */
    ComplexMatrix to_matrix =
        ComplexMatrix::Zero(1ULL << to_targets.size(), 1ULL << to_targets.size());
    for (UINT i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (UINT j : std::views::iota(0ULL, 1ULL << from_targets.size())) {
            for (UINT o : outer_indices) {
                to_matrix(idx_map[i] | o, idx_map[j] | o) = from_matrix(i, j);
            }
        }
    }
    /* std::cerr << "success" << std::endl; */
    return to_matrix;
}

inline std::optional<ComplexMatrix> get_pauli_matrix(PauliOperator pauli) {
    std::vector<UINT> pauli_id_list = pauli.get_pauli_id_list();
    UINT flip_mask, phase_mask, rot90_count;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, pauli_id_list.size()),
        [&](UINT i, UINT& f_mask, UINT& p_mask, UINT& rot90_cnt) {
            UINT pauli_id = pauli_id_list[i];
            if (pauli_id == 1) {
                f_mask += 1ULL << i;
            } else if (pauli_id == 2) {
                f_mask += 1ULL << i;
                p_mask += 1ULL << i;
                rot90_cnt++;
            } else if (pauli_id == 3) {
                p_mask += 1ULL << i;
            }
        },
        flip_mask,
        phase_mask,
        rot90_count);
    std::vector<StdComplex> rot = {1, -1.i, -1, 1.i};
    UINT matrix_dim = 1ULL << pauli_id_list.size();
    ComplexMatrix mat = ComplexMatrix::Zero(matrix_dim, matrix_dim);
    for (UINT index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1. - 2. * (Kokkos::popcount(index & phase_mask) % 2);
        mat(index, index ^ flip_mask) = rot[rot90_count % 4] * sign;
    }
    return mat;
}

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
inline Kokkos::View<T*> convert_host_vector_to_device_view(const std::vector<T>& vec) {
    Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        vec.data(), vec.size());
    Kokkos::View<T*> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T>
inline std::vector<T> convert_device_view_to_host_vector(const Kokkos::View<T*>& device_view) {
    std::vector<T> host_vector(device_view.extent(0));
    Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> host_view(
        host_vector.data(), host_vector.size());
    Kokkos::deep_copy(host_view, device_view);
    return host_vector;
}

// Device Kokkos::View を Host std::vector に変換する関数
template <typename T, typename Layout>
inline std::vector<std::vector<T>> convert_2d_device_view_to_host_vector(
    const Kokkos::View<T**, Layout>& view_d) {
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_d);
    std::vector<std::vector<T>> result(view_d.extent(0), std::vector<T>(view_d.extent(1), 0));
    for (size_t i = 0; i < view_d.extent(0); ++i) {
        for (size_t j = 0; j < view_d.extent(1); ++j) {
            result[i][j] = view_h(i, j);
        }
    }
    return result;
}

KOKKOS_INLINE_FUNCTION double squared_norm(const Complex& z) {
    return z.real() * z.real() + z.imag() * z.imag();
}

}  // namespace internal

}  // namespace scaluq
