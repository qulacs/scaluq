#pragma once

#include <Kokkos_Core.hpp>
#include <complex>
#include <cstdint>

namespace scaluq {

using InitializationSettings = Kokkos::InitializationSettings;

inline void initialize(const InitializationSettings& settings = InitializationSettings()) {
    Kokkos::initialize(settings);
}
inline void finalize() { Kokkos::finalize(); }

using UINT = std::uint64_t;

using Complex = Kokkos::complex<double>;
using namespace std::complex_literals;

struct array_4 {
    Complex val[4];
};

struct matrix_2_2 {
    Complex val[2][2];
};

struct matrix_4_4 {
    Complex val[4][4];
};

struct diagonal_matrix_2_2 {
    Complex val[2];
};

}  // namespace scaluq
