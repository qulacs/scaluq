#include <iostream>
#include <scaluq/all.hpp>

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        scaluq::internal::SparseMatrix<scaluq::Precision::F64, scaluq::ExecutionSpace::Default> mat(
            scaluq::SparseComplexMatrix(2, 2));
    }
    scaluq::finalize();
}
