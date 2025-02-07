#include <scaluq/state/state_vector.hpp>

int main() {
    constexpr scaluq::Precision Prec = scaluq::Precision::F64;
    scaluq::initialize();
    {
        scaluq::StateVector<Prec> state(2);
        std::cout << state << std::endl;
    }
    scaluq::finalize();
}
