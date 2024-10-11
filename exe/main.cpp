#include <Kokkos_Core.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gate/gate_factory.hpp"

// 使用例
int main() {
    Kokkos::initialize();
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<double> state(n_qubits);
        state.load({0, 1, 2, 3, 4, 5, 6, 7});
        auto x_gate = scaluq::gate::X<double>(1, {0, 2});
        x_gate->update_quantum_state(state);
        auto sqrtx_gate = scaluq::gate::SqrtX<double>(1, {0});
        sqrtx_gate->update_quantum_state(state);
        auto sqrtxdag_gate = scaluq::gate::SqrtXdag<double>(0);
        sqrtxdag_gate->update_quantum_state(state);

        std::cout << state << std::endl;
    }
    Kokkos::finalize();
}
