#include <scaluq/state/state_vector.hpp>

int main() {
    scaluq::initialize();
    {
        scaluq::StateVector<double> state(2);
        std::cout << state << std::endl;
    }
    scaluq::finalize();
}
