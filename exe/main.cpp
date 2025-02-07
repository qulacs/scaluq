#include <cstdint>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

int main() {
    constexpr Precision Prec = Precision::F64;
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<Prec> state(n_qubits);
        state.load({0, 1, 2, 3, 4, 5, 6, 7});
        Json j = state;
        std::cout << j << std::endl;
        state = j;
        std::cout << state << std::endl;

        scaluq::Circuit<Prec> circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X<Prec>(0));
        circuit.add_gate(scaluq::gate::CNot<Prec>(0, 1));
        circuit.add_gate(scaluq::gate::Y<Prec>(1));
        circuit.add_gate(scaluq::gate::RX<Prec>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        scaluq::Operator<Prec> observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    {
        std::uint64_t n_qubits = 2, batch_size = 2;
        scaluq::StateVectorBatched<Prec> states(batch_size, n_qubits);
        states.set_Haar_random_state(batch_size, n_qubits, false);
        Json j = states;
        std::cout << j << std::endl;
        states = j;
        std::cout << states << std::endl;
    }
    {
        double coef = 2.0;
        std::string pauli_string = "X 0 Z 1 Y 2";
        PauliOperator<Prec> pauli(pauli_string, coef);
        Json j = pauli;
        std::cout << j << std::endl;
        pauli = j;
    }
    {
        std::uint64_t n_qubits = 3;
        Operator<Prec> op(n_qubits);
        op.add_operator({0b001, 0b010, 2});
        op.add_operator({"X 2 Y 1", 1});
        Json j = op;
        std::cout << j << std::endl;
        op = j;
    }
    {
        std::cout << Json(gate::I<Prec>()) << std::endl;
        std::cout << Json(gate::X<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Y<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Z<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::H<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::S<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Sdag<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::T<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::Tdag<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtX<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtXdag<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtY<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::SqrtYdag<Prec>(2, {0, 3})) << std::endl;
        std::cout << Json(gate::RX<Prec>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::RY<Prec>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::RZ<Prec>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::U1<Prec>(2, 0.5, {0, 3})) << std::endl;
        std::cout << Json(gate::U2<Prec>(2, 0.5, 0.3, {0, 3})) << std::endl;
        std::cout << Json(gate::U3<Prec>(2, 0.5, 0.3, 0.1, {0, 3})) << std::endl;
        std::cout << Json(gate::Swap<Prec>(1, 2, {0, 3})) << std::endl;

        PauliOperator<Prec> pauli("X 2 Y 1");
        std::cout << Json(gate::Pauli<Prec>(pauli)) << std::endl;
        std::cout << Json(gate::PauliRotation<Prec>(pauli, 0.5)) << std::endl;

        auto probgate = gate::Probablistic<Prec>({.1, .9}, {gate::X<Prec>(0), gate::I<Prec>()});
        std::cout << Json(probgate) << std::endl;

        std::cout << Json(gate::ParamRX<Prec>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamRY<Prec>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamRZ<Prec>(2, 1.5, {0, 3})) << std::endl;
        std::cout << Json(gate::ParamPauliRotation<Prec>(pauli, 0.5)) << std::endl;

        auto paramprobgate =
            gate::ParamProbablistic<Prec>({.1, .9}, {gate::ParamRX<Prec>(0), gate::I<Prec>()});
        std::cout << Json(paramprobgate) << std::endl;
    }
    {
        auto x = gate::X<Prec>(1, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::RX<Prec>(1, 0.5, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::Swap<Prec>(1, 3, {2});
        Json j = x;
        std::cout << j << std::endl;
        Gate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        PauliOperator<Prec> pauli("X 2 Y 1");
        auto x = gate::PauliRotation<Prec>(pauli, 1.5, {0, 3});
        Json j = x;
        std::cout << j << std::endl;
        Gate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto probgate =
            gate::Probablistic<Prec>({.1, .9}, {gate::X<Prec>(0, {2, 3}), gate::I<Prec>()});
        Json j = probgate;
        std::cout << j << std::endl;
        Gate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto x = gate::ParamRX<Prec>(1, {2});
        Json j = x;
        std::cout << j << std::endl;
        ParamGate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        auto paramprobgate =
            gate::ParamProbablistic<Prec>({.1, .9}, {gate::ParamRX<Prec>(0), gate::I<Prec>()});
        Json j = paramprobgate;
        std::cout << j << std::endl;
        ParamGate<Prec> gate = j;
        std::cout << gate << std::endl;
    }
    {
        Circuit<Prec> circuit(10);
        circuit.add_gate(gate::X<Prec>(0, {3}));
        circuit.add_param_gate(gate::ParamRX<Prec>(0, 0.5, {3}), "RX");
        Json j = circuit;
        std::cout << j << std::endl;
        Circuit<Prec> c = j;
        std::cout << Json(c) << std::endl;
    }

    Kokkos::finalize();
}
