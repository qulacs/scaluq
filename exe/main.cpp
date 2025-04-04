#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;
using namespace nlohmann;

constexpr auto F64 = Precision::F64;
constexpr auto GPU = ExecutionSpace::Default;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 5;
        const std::uint64_t omega = 3;

        // std::vector<std::uint64_t> U_omega_control;
        // for (int i = 0; i < n_qubits; ++i) {
        //     if (i != omega) U_omega_control.push_back(i);
        // }
        // auto U_w = gate::Z<F64, GPU>(n_qubits, U_omega_control);
        // std::cout << U_w << std::endl;
        // std::cout << U_w->control_value_mask() << std::endl;
        // auto U_w_cv = U_w->control_value_list();
        // std::cout << U_w_cv.size() << std::endl;
        // for (auto cv : U_w_cv) {
        //     std::cout << cv << " ";
        // }
        // std::cout << std::endl;

        Circuit<F64, GPU> U_s(n_qubits);
        for (int i = 0; i < n_qubits; ++i) {
            U_s.add_gate(gate::H<F64, GPU>(i));
        }

        // std::vector<std::uint64_t> U_s_control, U_s_cvalue;
        // for (int i = 0; i < n_qubits; ++i) {
        //     U_s_control.push_back(i);
        //     U_s_cvalue.push_back(0);
        // }
        // U_s.add_gate(gate::GlobalPhase<F64, GPU>(M_PI, U_s_control, U_s_cvalue));  // 位相を -1
        // for (int i = 0; i < n_qubits; ++i) U_s.add_gate(gate::H<F64, GPU>(i));

        // Circuit<F64, GPU> Grover(n_qubits);
        // for (int i = 0; i < n_qubits; ++i) Grover.add_gate(gate::H<F64, GPU>(i));
        // for (int i = 0; i < 1; ++i) {
        //     Grover.add_gate(U_w);
        //     Grover.add_circuit(U_s);
        // }
        // StateVector<F64, GPU> state(n_qubits);
        // Grover.update_quantum_state(state);
        // std::cout << state << std::endl;
    }
    scaluq::finalize();
}
