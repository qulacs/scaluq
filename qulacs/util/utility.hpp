#include <Kokkos_Core.hpp>
#include <algorithm>  // For std::copy
#include <iostream>
#include <vector>

#include "../types.hpp"

// Host std::vector を Device Kokkos::View に変換する関数
template <typename T>
Kokkos::View<T*, Kokkos::DefaultExecutionSpace> convert_host_vector_to_device_view(
    const std::vector<T>& vec) {
    Kokkos::View<T*, Kokkos::HostSpace> host_view("host_view", vec.size());
    std::copy(vec.begin(), vec.end(), host_view.data());
    Kokkos::View<T*, Kokkos::DefaultExecutionSpace> device_view("device_view", vec.size());
    Kokkos::deep_copy(device_view, host_view);
    return device_view;
}
