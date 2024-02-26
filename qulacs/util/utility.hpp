#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <vector>

#include "../types.hpp"

namespace qulacs {

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

}  // namespace internal

KOKKOS_INLINE_FUNCTION double norm2(const Complex& z) {
    return z.real() * z.real() + z.imag() * z.imag();
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

};  // namespace qulacs
