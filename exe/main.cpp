#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        scaluq::Gate<scaluq::Precision::F64, scaluq::ExecutionSpace::Host> g =
            scaluq::gate::H<scaluq::Precision::F64, scaluq::ExecutionSpace::Host>(0);
        scaluq::Json j;
        scaluq::Gate g2 =
            j.template get<scaluq::Gate<scaluq::Precision::F64, scaluq::ExecutionSpace::Host>>();
        std::cout << g2->to_string() << std::endl;
    }
    scaluq::finalize();
}
