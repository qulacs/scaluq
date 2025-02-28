#include <scaluq/state/state_vector.hpp>

int main() {
    constexpr scaluq::Precision Prec = scaluq::Precision::F64;
    constexpr scaluq::Space Space = scaluq::Space::Default;
    scaluq::initialize();
    {
        scaluq::StateVector<Prec, Default> state(2);
        std::cout << state << std::endl;
    }
    scaluq::finalize();
}
