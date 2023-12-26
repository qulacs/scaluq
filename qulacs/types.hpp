#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace qulacs {
using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;

struct matrix_2_2 {
    Complex val[2][2];
};

}  // namespace qulacs
