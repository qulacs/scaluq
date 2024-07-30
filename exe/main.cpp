#include <Eigen/Core>
#include <functional>
#include <iostream>

#include "../scaluq/all.hpp"

using namespace scaluq;
using namespace std;

std::int64_t run(UINT n, UINT t, bool gemm) {
    std::vector<UINT> targets(t);
    Random random;
    for (UINT i : std::views::iota(0ULL, t)) {
        targets[i] = random.int32() % (n - i);
        for (UINT j : std::views::iota(0ULL, i)) {
            if (targets[i] == targets[j]) targets[i] = n - 1 - j;
        }
    }
    UINT matrix_dim = 1ULL << t;
    ComplexMatrix mat(matrix_dim, matrix_dim);
    for (UINT i : std::views::iota(0ULL, matrix_dim)) {
        for (UINT j : std::views::iota(0ULL, matrix_dim)) {
            mat(i, j) = StdComplex(random.uniform(), random.uniform());
        }
    }
    auto g = gate::DenseMatrix(mat, targets);
    auto dg = DenseMatrixGate(g);
    StateVector state = StateVector::Haar_random_state(n);
    auto st = std::chrono::system_clock::now();
    if (gemm) {
        dg->update_quantum_state_gemm(state);
    } else {
        dg->update_quantum_state(state);
    }
    auto ed = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count();
}

int main() {
    Kokkos::initialize();
    UINT t = 3;
    for (UINT n : std::views::iota(t, 26ULL)) {
        UINT count = 10;
        std::int64_t sum_time_false = 0, sum_time_true = 0;
        for (auto _ : std::views::iota(0ULL, count)) {
            sum_time_false += run(n, t, false);
            sum_time_false += run(n, t, true);
        }
        std::cout << n << " " << sum_time_false / count << " " << sum_time_true / count
                  << std::endl;
    }
    Kokkos::finalize();
}
