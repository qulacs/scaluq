#include <Eigen/Core>
#include <functional>
#include <iostream>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() { return; }

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
