#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

using namespace scaluq;
using namespace std;

void run() {
    UINT n_qubits = 5;
    auto state = StateVector<>::Haar_random_state(n_qubits);
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
