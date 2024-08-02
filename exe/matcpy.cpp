#include <Eigen/Core>
#include <functional>
#include <iostream>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

void run() {
    ComplexMatrix cmp(2, 2);
    cmp(0, 0) = {0, 1};
    cmp(0, 1) = {2, 3};
    cmp(1, 0) = {4, 5};
    cmp(1, 1) = {6, 7};
    Matrix view = internal::convert_external_matrix_to_internal_matrix(cmp);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {2, 2}), KOKKOS_LAMBDA(int i, int j) {
            Kokkos::printf("(%d,%d) = (%f,%f)\n", i, j, view(i, j).real(), view(i, j).imag());
        });
    ComplexMatrix cmp2 = internal::convert_internal_matrix_to_external_matrix(view);
    std::cout << cmp2(0, 0).real() << ' ' << cmp2(0, 0).imag() << std::endl;
    std::cout << cmp2(0, 1).real() << ' ' << cmp2(0, 1).imag() << std::endl;
    std::cout << cmp2(1, 0).real() << ' ' << cmp2(1, 0).imag() << std::endl;
    std::cout << cmp2(1, 1).real() << ' ' << cmp2(1, 1).imag() << std::endl;
}

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
