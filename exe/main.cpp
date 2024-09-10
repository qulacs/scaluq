#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/state/state_vector.hpp"
#include "../scaluq/state/state_vector_batched.hpp"

using namespace scaluq;
using namespace std;

void run() {
    std::uint64_t n_qubits = 5;
    auto state = StateVector<>::Haar_random_state(n_qubits);
    auto state_b = StateVectorBatched<>::Haar_random_states(2, 2, true);
    std::cout << state << std::endl;
    std::cout << state_b << std::endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
