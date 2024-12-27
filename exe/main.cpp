#include <cstdint>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<double> state(n_qubits);
        state.load({0, 1, 2, 3, 4, 5, 6, 7});
        Json j = state;
        std::cout << j << std::endl;
        state = j;
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
    {
        std::uint64_t n_qubits = 2, batch_size = 2;
        scaluq::StateVectorBatched<double> states(batch_size, n_qubits);
        states.set_Haar_random_state(batch_size, n_qubits, false);
        Json j = states;
        std::cout << j << std::endl;
        states = j;
        std::cout << states << std::endl;
    }
    {
        double coef = 2.0;
        std::string pauli_string = "X 0 Z 1 Y 2";
        PauliOperator<double> pauli(pauli_string, coef);
        Json j = pauli;
        std::cout << j << std::endl;
        pauli = j;
    }
    {
        std::uint64_t n_qubits = 3;
        Operator<double> op(n_qubits);
        op.add_operator({0b001, 0b010, Complex<double>(2)});
        op.add_operator({"X 2 Y 1", 1});
        Json j = op;
        std::cout << j << std::endl;
        op = j;
    }
    {
        std::cout << Json(gate::I<double>()) << std::endl;
        std::cout << Json(gate::X<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Y<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Z<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::H<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::S<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Sdag<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::T<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Tdag<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtX<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtXdag<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtY<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtYdag<double>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::RX<double>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::RY<double>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::RZ<double>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::U1<double>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::U2<double>(2, 0.5, 0.3, {0, 3})) << std::endl;
        std::cout << Json(gate::U3<double>(2, 0.5, 0.3, 0.1, {0, 3})) << std::endl;
        std::cout << Json(gate::Swap<double>(1, 2, {0, 3})) << std::endl;

        PauliOperator<double> pauli("X 2 Y 1");
        std::cout << Json(gate::Pauli<double>(pauli)) << std::endl;
        std::cout << Json(gate::PauliRotation<double>(pauli, 0.5)) << std::endl;

        auto probgate =
            gate::Probablistic<double>({.1, .9}, {gate::X<double>(0), gate::I<double>()});
        std::cout << Json(probgate) << std::endl;

        std::cout << Json(gate::ParamRX<double>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamRY<double>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamRZ<double>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamPauliRotation<double>(pauli, 0.5)) << std::endl;

        auto paramprobgate = gate::ParamProbablistic<double>(
            {.1, .9}, {gate::ParamRX<double>(0), gate::I<double>()});
        std::cout << Json(paramprobgate) << std::endl;
    }
    {
        auto x = gate::X<double>(1, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::RX<double>(1, 0.5, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::Swap<double>(1, 3, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        PauliOperator<double> pauli("X 2 Y 1");
        auto x = gate::PauliRotation<double>(pauli, 1.5, {0, 3});
        Json j = x;
        std::cout << j << std::endl;
        Gate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto probgate =
            gate::Probablistic<double>({.1, .9}, {gate::X<double>(0, {2, 3}), gate::I<double>()});
        Json j = probgate;
        std::cout << j << std::endl;
        Gate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::ParamRX<double>(1, {2});
        Json j = x;
        std::cout << j << std::endl;
        ParamGate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto paramprobgate = gate::ParamProbablistic<double>(
            {.1, .9}, {gate::ParamRX<double>(0), gate::I<double>()});
        Json j = paramprobgate;
        std::cout << j << std::endl;
        ParamGate<double> gate = j;
        std::cout << gate << std::endl;
    }
    {
        Circuit<double> circuit(10);
        circuit.add_gate(gate::X<double>(0, {3}));
        circuit.add_param_gate(gate::ParamRX<double>(0, 0.5, {3}), "RX");
        Json j = circuit;
        std::cout << j << std::endl;
        Circuit<double> c = j;
        std::cout << Json(c) << std::endl;
    }

    Kokkos::finalize();

    std::cout << "Type of _Float64: " << typeid(_Float64).name() << std::endl;
    std::cout << "Type of double: " << typeid(double).name() << std::endl;
}
