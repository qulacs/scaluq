#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>

// #include "../scaluq/gate/gate_factory.hpp"
#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() { return; }

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
