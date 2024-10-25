#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() {
    StateVector state = StateVector::Haar_random_state(3);
    Json j = state;
    cout << j.dump() << endl;
    StateVector state2 = j.get<StateVector>();
    cout << state2 << endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
