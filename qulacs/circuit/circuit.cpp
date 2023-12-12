#include "circuit.hpp"

#include <Kokkos_Vector.hpp>

namespace qulacs {
void Circuit::add_gate(Gate gate) {
    check_gate_is_valid(*gate);
    _gate_list.push_back(std::move(gate));
}

void Circuit::update_quantum_state(StateVector& state) const {
    for (const auto& gate_ptr : _gate_list) {
        gate_ptr->update_quantum_state(state);
    }
}

Circuit Circuit::copy() const {
    Circuit ccircuit(_n_qubits);
    for (auto&& gate_ptr : _gate_list) {
        ccircuit.add_gate(gate_ptr->copy());
    }
    return ccircuit;
}

Circuit Circuit::get_inverse() const {
    Circuit icircuit(_n_qubits);
    for (auto&& gate_ptr : _gate_list | std::views::reverse) {
        icircuit.add_gate(gate_ptr->get_inverse());
    }
    return icircuit;
}

void Circuit::check_gate_is_valid(const QuantumGate& gate) const {
    auto targets = gate.get_target_qubit_list();
    auto controls = gate.get_control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}
}  // namespace qulacs
