#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"
#include "../scaluq/util/utility.hpp"

using namespace scaluq;
using namespace std;

void run() {
    std::uint64_t n_qubits = 5;
    Kokkos::parallel_for(
        10, KOKKOS_LAMBDA(int i) {
            Kokkos::printf("{%lf, %lf, %lf, %lf}\n",
                           EXAMPLE(0).real(),
                           EXAMPLE(1).real(),
                           EXAMPLE(2).real(),
                           EXAMPLE(3).real());
            Kokkos::printf("%lf\n", INVERSE_SQRT2);
        });
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
