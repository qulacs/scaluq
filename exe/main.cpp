#include <Kokkos_Core.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <scaluq/state/state_vector.hpp>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "scaluq/all.hpp"

using namespace scaluq;
using namespace std;

// void run() {
//     StateVector state = StateVector<double>::Haar_random_state(3);
//     Json j = state;
//     cout << j.dump() << endl;
//     StateVector state2 = j.get<StateVector>();
//     cout << state2 << endl;
// }

int main() {
    Kokkos::initialize();
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<double> state(n_qubits);
        state.load({0, 1, 2, 3, 4, 5, 6, 7});
        Json j = state;
        std::cout << j.dump(2) << std::endl;
        state = j;
        std::cout << state << std::endl;
    }
    {
        std::uint64_t n_qubits = 2, batch_size = 2;
        scaluq::StateVectorBatched<double> states(batch_size, n_qubits);
        states.set_Haar_random_state(batch_size, n_qubits, false);
        Json j = states;
        std::cout << j.dump(2) << std::endl;
        states = j;
        std::cout << states << std::endl;
    }
    {
        double coef = 2.0;
        std::string pauli_string = "X 0 Z 1 Y 2";
        PauliOperator<double> pauli(pauli_string, coef);
        Json j = pauli;
        std::cout << j.dump(2) << std::endl;
        pauli = j;
    }
    {
        std::uint64_t n_qubits = 3;
        Operator<double> op(n_qubits);
        op.add_operator({0b001, 0b010, Complex<double>(2)});
        op.add_operator({"X 2 Y 1", 1});
        Json j = op;
        std::cout << j.dump(2) << std::endl;
        op = j;
    }
    // {
    //     XGate<double> x = gate::X<double>(2);
    //     Json j = x;
    //     std::cout << j << std::endl;
    // }

    Kokkos::finalize();
}
