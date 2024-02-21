#include <iostream>

#include "state/state_vector.hpp"

using namespace qulacs;

void run() {
    auto state = StateVector::Haar_random_state(3);
    for (UINT i = 0; i < state.dim(); i++) {
        // std::cout << state[i] << std::endl;
    }
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
