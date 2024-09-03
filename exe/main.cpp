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
    auto state = StateVector::Haar_random_state(n_qubits);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
