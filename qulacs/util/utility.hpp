#include <Kokkos_Core.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <vector>

#include "../types.hpp"

namespace qulacs {

namespace internal {

/**
 * Insert 0 to insert_index-th bit of basis_index.
 */
KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
    UINT mask = (1ULL << insert_index) - 1;
    UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
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

#define _CHECK_GT(val1, val2) _check_gt(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
static std::string _check_gt(
    T val1, T val2, std::string val1_name, std::string val2_name, std::string file, UINT line) {
    if (val1 > val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") > (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

inline static UINT count_population(UINT x) {
    x = ((x & 0xaaaaaaaaaaaaaaaaUL) >> 1) + (x & 0x5555555555555555UL);
    x = ((x & 0xccccccccccccccccUL) >> 2) + (x & 0x3333333333333333UL);
    x = ((x & 0xf0f0f0f0f0f0f0f0UL) >> 4) + (x & 0x0f0f0f0f0f0f0f0fUL);
    x = ((x & 0xff00ff00ff00ff00UL) >> 8) + (x & 0x00ff00ff00ff00ffUL);
    x = ((x & 0xffff0000ffff0000UL) >> 16) + (x & 0x0000ffff0000ffffUL);
    x = ((x & 0xffffffff00000000UL) >> 32) + (x & 0x00000000ffffffffUL);
    return (UINT)x;
}
}  // namespace qulacs
