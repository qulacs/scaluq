#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() {
    internal::SparseComplexMatrix sparse(2, 2);
    sparse.insert(0, 0) = 2;
    auto g = gate::SparseMatrix({0}, sparse);
    auto mat = g->get_matrix();
    std::cout << mat << endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
