#include <bitset>
#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;
using namespace nlohmann;

constexpr auto F64 = Precision::F64;
constexpr auto GPU = ExecutionSpace::Default;

Gate<F64, GPU> get_Uw(std::uint64_t n_qubits, std::uint64_t omega) {
    std::vector<std::uint64_t> U_omega_control, U_omega_cvalue;
    for (int i = 0; i < n_qubits; ++i) {
        U_omega_control.push_back(i);
        U_omega_cvalue.push_back((omega >> i) & 1);
    }
    return gate::GlobalPhase<F64, GPU>(M_PI, U_omega_control, U_omega_cvalue);
}

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        Kokkos::Timer timer;

        constexpr std::uint64_t n_qubits = 15;
        std::uint64_t w = rand() % (1 << n_qubits);  // 解のインデックス

        Circuit<F64, GPU> Loop(n_qubits);
        Loop.add_gate(get_Uw(n_qubits, w));

        // get_Us
        for (int i = 0; i < n_qubits; ++i) Loop.add_gate(gate::H<F64, GPU>(i));
        std::vector<std::uint64_t> Us_control, Us_cvalue;  // 2|0^n><0^n| - I を作る
        for (int i = 0; i < n_qubits; ++i) {
            Us_control.push_back(i);
            Us_cvalue.push_back(0);
        }
        Loop.add_gate(gate::GlobalPhase<F64, GPU>(M_PI, Us_control, Us_cvalue));  // 位相を -1
        for (int i = 0; i < n_qubits; ++i) Loop.add_gate(gate::H<F64, GPU>(i));

        // loop Uw, Us
        std::uint64_t loop_max = M_PI * std::sqrt(1 << n_qubits) / 4;
        Circuit<F64, GPU> Grover(n_qubits);
        for (int i = 0; i < n_qubits; ++i) Grover.add_gate(gate::H<F64, GPU>(i));
        for (int i = 0; i < loop_max; ++i) Grover.add_circuit(Loop);

        StateVector<F64, GPU> state(n_qubits);
        Grover.update_quantum_state(state);
        auto vec = state.get_amplitudes();
        std::cout << "Amplitude of |" << std::bitset<n_qubits>(w) << "> : " << vec[w] << std::endl;

        std::cout << "Time: " << timer.seconds() << std::endl;
    }
    {
        std::cout << gate::X<F64, GPU>(3, {1, 2}, {0, 1}) << std::endl;
        std::cout << Json(gate::X<F64, GPU>(3, {1, 2}, {0, 1})) << std::endl;

        StateVector<F64, GPU> state(2);
        state.load({1, 2, 3, 4});
        auto cx = gate::CX<F64, GPU>(1, 0);
        cx->update_quantum_state(state);
        std::cout << state << std::endl;
    }
    scaluq::finalize();
}
