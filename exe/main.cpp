#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    std::vector<PauliOperator<Prec, Space>> terms = {PauliOperator<Prec, Space>("X 0 Y 1", 1.0),
                                                     PauliOperator<Prec, Space>("Z 0", 2.0),
                                                     PauliOperator<Prec, Space>("Y 1", 3.0)};
    Operator<Prec, Space> op(terms);
    Json j;
    to_json(j, op);
    std::cout << "Serialized Operator: " << j.dump(4) << std::endl;
    Operator<Prec, Space> op2;
    from_json(j, op2);
    std::cout << "Deserialized Operator: " << op2.to_string() << std::endl;
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        constexpr Precision Prec = scaluq::Precision::F64;
        constexpr ExecutionSpace Space = scaluq::ExecutionSpace::Default;
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector<Prec, Space> state =
            scaluq::StateVector<Prec, Space>::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit<Prec, Space> circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X<Prec, Space>(0));
        circuit.add_gate(scaluq::gate::CNot<Prec, Space>(0, 1));
        circuit.add_gate(scaluq::gate::Y<Prec, Space>(1));
        circuit.add_gate(scaluq::gate::RX<Prec, Space>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        std::vector<scaluq::PauliOperator<Prec, Space>> paulis;
        paulis.emplace_back(1, 0);
        scaluq::Operator<Prec, Space> observable(paulis);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
