#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() {
    auto y_gate = gate::Y(2);
    std::cout << y_gate << std::endl;

    auto cx_gate = gate::CX(0, 2);
    std::cout << cx_gate << std::endl;

    auto swap_gate = gate::Swap(2, 3, {4, 6});
    std::cout << swap_gate << "\n\n";

    auto prob_gate = gate::Probablistic({0.1, 0.1, 0.8}, {cx_gate, y_gate, swap_gate});
    std::cout << prob_gate << "\n\n";

    auto prob_prob_gate = gate::Probablistic({0.5, 0.5}, {cx_gate, prob_gate});
    std::cout << prob_prob_gate << "\n\n";

    auto prx_gate = gate::ParamRX(2);
    std::cout << prx_gate << "\n\n";

    auto pry_gate = gate::ParamRY(2, 2.5, {1, 3});
    std::cout << pry_gate << "\n\n";

    auto pprob_gate = gate::ParamProbablistic({0.7, 0.3}, {prx_gate, pry_gate});
    std::cout << pprob_gate << std::endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
