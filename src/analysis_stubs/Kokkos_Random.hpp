#pragma once
#ifndef KOKKOS_RANDOM_HPP
#define KOKKOS_RANDOM_HPP
#endif
// Minimal Kokkos_Random stub for clang-tidy static analysis.

namespace Kokkos {

template<class ExecSpace>
class Random_XorShift64_Pool {
public:
    struct generator_type {
        template<class T = double>
        T normal(T = T{}, T = T{1}) { return {}; }
        template<class T = double>
        T drand(T = T{}, T = T{1}) { return {}; }
        unsigned long long urand64() { return 0; }
    };

    explicit Random_XorShift64_Pool(unsigned long long = 0) {}
    generator_type get_state()  const         { return {}; }
    void           free_state(generator_type) const {}
};

} // namespace Kokkos
