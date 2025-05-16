#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;

using namespace scaluq;

int main() {
    initialize();  // must be called before using any scaluq methods
    {
        {       
            std::int64_t n = 10;
            std::int64_t d = 1 << n;
            std::int64_t m = n / 2;
            std::int64_t md = 1 << m;
            std::int32_t b = 100;
            std::int32_t r = 500;
            Random rd;

            ComplexMatrix matrix(md, md);
            for (int i = 0; i < md; ++i) {
                for (int j = 0; j < md; ++j) {
                    matrix(i, j) = StdComplex(rd.normal(), rd.normal());
                }
            }
            std::vector<std::uint64_t> targets = [&] {
                auto tmp = rd.permutation(n);
                std::vector<std::uint64_t> prefix;
                for (int i = 0; i < m; ++i) prefix.push_back(tmp[i]);
                return prefix;
            }();

            auto mgate = gate::DenseMatrix<Precision::F64,ExecutionSpace::Default>(targets, matrix);

            StateVector<Precision::F64,ExecutionSpace::Default> state(n);
            StateVectorBatched<Precision::F64,ExecutionSpace::Default> batched_state(b, n);
            for (int i = 0; i < r; ++i) {
                for (int j = 0; j < b; ++j) {
                    mgate->update_quantum_state(state);
                }
            }
            for (int i = 0; i < r; ++i) {
                mgate->update_quantum_state(batched_state);
            }
        }
    }
    finalize();
}
