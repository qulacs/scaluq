#include <iostream>

#include "operator/operator.hpp"

using namespace qulacs;

void run() {
    auto op1 = Operator(2);
    op1.add_random_operator(2);
    auto op2 = Operator(2);
    op2.add_random_operator(2);
    std::cerr << op1.to_string() << std::endl;
    std::cerr << op2.to_string() << std::endl;
    std::cerr << (op1 * op2).to_string() << std::endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
