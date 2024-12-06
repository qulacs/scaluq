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
        /*
        auto x_gate = scaluq::gate::X<double>(1, {0, 2});
        x_gate->update_quantum_state(state);
        auto sqrtx_gate = scaluq::gate::SqrtX<double>(1, {0});
        sqrtx_gate->update_quantum_state(state);
        auto sqrtxdag_gate = scaluq::gate::SqrtXdag<double>(0);
        sqrtxdag_gate->update_quantum_state(state);
        */

        std::cout << state << std::endl;
    }
    {
        std::uint64_t n_qubits = 3;
        scaluq::StateVector<double> state(n_qubits);
        Json j;
        to_json(j, state);
        std::cout << j.dump(2) << std::endl;
    }
    {
        std::uint64_t n_qubits = 2, batch_size = 2;
        scaluq::StateVectorBatched<double> states(batch_size, n_qubits);
        Json j;
        to_json(j, states);
        std::cout << j.dump(2) << std::endl;
    }

    Kokkos::finalize();
}
