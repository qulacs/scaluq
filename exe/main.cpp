#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;

int main() {
    initialize();  // must be called before using any scaluq methods
    {
        Kokkos::Timer timer;
        {       
            std::int64_t n = 16;
            std::int64_t d = 1 << n;
            std::int64_t m = n / 2;
            std::int64_t md = 1 << m;
            std::int32_t b = 100;
            std::int32_t r = 500;
            Random rd;

            SparseComplexMatrix sp(md, md);
            std::vector<Eigen::Triplet<StdComplex>> trp;
            for (int i = 0; i < md; ++i) {
                for (int j = 0; j < md; ++j) {
                    if (rand() % 10 == 0) {
                        trp.emplace_back(i, j, StdComplex(rd.normal(), rd.normal()));
                    }
                }
            }
            sp.setFromTriplets(trp.begin(), trp.end());
            std::cout << sp.nonZeros() << std::endl;

            std::vector<std::uint64_t> targets = [&] {
                auto tmp = rd.permutation(n);
                std::vector<std::uint64_t> prefix;
                for (int i = 0; i < m; ++i) prefix.push_back(tmp[i]);
                return prefix;
            }();

            auto mgate = gate::SparseMatrix<Precision::F64,ExecutionSpace::Default>(targets, sp);

            StateVector<Precision::F64,ExecutionSpace::Default> state(n);
            for (int i = 0; i < r; ++i) {
                    mgate->update_quantum_state(state);
            }
        }
        auto d = timer.seconds();
        std::cout << d << std::endl;
    }
    finalize();
}
