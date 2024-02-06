#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace qulacs {

inline void initialize() { Kokkos::initialize(); }
inline void finalize() { Kokkos::finalize(); }

using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;

struct matrix_2_2 {
    Complex val[2][2];
};

struct diagonal_matrix_2_2 {
    Complex val[2];
};

}  // namespace qulacs
