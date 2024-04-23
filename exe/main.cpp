#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>
#include <util/random.hpp>

#include "../scaluq/all.hpp"
#include "../scaluq/util/utility.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

using namespace scaluq;
using namespace std;

void run() { return; }

int main() {
    Kokkos::initialize();
    run();
    Kokkos::finalize();
}
