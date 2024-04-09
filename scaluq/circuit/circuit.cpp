#include "circuit.hpp"

#include <ranges>

namespace scaluq {
UINT Circuit::calculate_depth() const {
    std::vector<UINT> filled_step(_n_qubits, 0ULL);
    for (const auto& gate : _gate_list) {
        std::vector<UINT> control_qubits = gate->get_control_qubit_list();
        std::vector<UINT> target_qubits = gate->get_control_qubit_list();
        UINT max_step_amount_target_qubits = 0;
        for (UINT control : control_qubits) {
            if (max_step_amount_target_qubits < filled_step[control]) {
                max_step_amount_target_qubits = filled_step[control];
            }
        }
        for (UINT target : control_qubits) {
            if (max_step_amount_target_qubits < filled_step[target]) {
                max_step_amount_target_qubits = filled_step[target];
            }
        }
        for (UINT control : control_qubits) {
            filled_step[control] = max_step_amount_target_qubits + 1;
        }
        for (UINT target : target_qubits) {
            filled_step[target] = max_step_amount_target_qubits + 1;
        }
    }
    return *std::ranges::max_element(filled_step);
}

void Circuit::add_gate(const Gate& gate) {
    check_gate_is_valid(gate);
    _gate_list.push_back(gate->copy());
}
void Circuit::add_gate(Gate&& gate) {
    check_gate_is_valid(gate);
    _gate_list.push_back(std::move(gate));
}
void Circuit::add_circuit(const Circuit& circuit) {
    for (const auto& gate : circuit._gate_list) {
        add_gate(gate);
    }
}
void Circuit::add_circuit(Circuit&& circuit) {
    for (auto&& gate : circuit._gate_list) {
        add_gate(std::move(gate));
    }
}

void Circuit::update_quantum_state(StateVector& state) const {
    for (const auto& gate : _gate_list) {
        gate->update_quantum_state(state);
    }
}

Circuit Circuit::copy() const {
    Circuit ccircuit(_n_qubits);
    for (const auto& gate : _gate_list) {
        ccircuit.add_gate(gate->copy());
    }
    return ccircuit;
}

Circuit Circuit::get_inverse() const {
    Circuit icircuit(_n_qubits);
    for (const auto& gate : _gate_list | std::views::reverse) {
        icircuit.add_gate(gate->get_inverse());
    }
    return icircuit;
}

void Circuit::check_gate_is_valid(const Gate& gate) const {
    auto targets = gate->get_target_qubit_list();
    auto controls = gate->get_control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}
}  // namespace scaluq
