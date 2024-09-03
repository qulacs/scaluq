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
    auto x_gate = gate::X(2);
    std::cout << x_gate << std::endl;
    auto y_gate = gate::Y(2);
    std::cout << y_gate << std::endl;
    auto swap_gate = gate::Swap(2, 3, {4, 6});
    std::cout << swap_gate << "\n\n";

    auto prob_gate = gate::Probablistic({0.1, 0.1, 0.8}, {x_gate, y_gate, swap_gate});
    std::cout << prob_gate << "\n\n";

    auto prob_prob_gate = gate::Probablistic({0.5, 0.5}, {x_gate, prob_gate});
    std::cout << prob_prob_gate << "\n\n";
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
