#include <scaluq/state/state_vector.hpp>

int main() {
    constexpr scaluq::Precision Prec = scaluq::Precision::F64;
    constexpr scaluq::ExecutionSpace Space = scaluq::ExecutionSpace::Default;
    scaluq::initialize();
    {
        scaluq::StateVector<Prec, Space> state(2);
        std::cout << state << std::endl;
    }
    scaluq::finalize();
}
