#include <cstdint>
#include <iostream>
#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/operator/operator.hpp>
#include <scaluq/state/state_vector.hpp>

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector state = scaluq::StateVector<double>::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit<double> circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X<double>(0));
        circuit.add_gate(scaluq::gate::CNot<double>(0, 1));
        circuit.add_gate(scaluq::gate::Y<double>(1));
        circuit.add_gate(scaluq::gate::RX<double>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        scaluq::Operator<double> observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
