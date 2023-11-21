#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace qulacs {
using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;
}  // namespace qulacs
