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
    if (gate.is_parametric()) {
        throw std::runtime_error(
            "Circuit::add_gate(const Gate&): parametric gate cannot add without parameter_key.");
    }
    _gate_list.push_back(gate->copy());
    _parameter_key_list.push_back("");
}
void Circuit::add_gate(Gate&& gate) {
    check_gate_is_valid(gate);
    if (gate.is_parametric()) {
        throw std::runtime_error(
            "Circuit::add_gate(const Gate&): parametric gate cannot add without parameter_key.");
    }
    _gate_list.push_back(std::move(gate));
    _parameter_key_list.push_back("");
}
void Circuit::add_gate(const Gate& gate, std::string_view parameter_key) {
    if (parameter_key.empty()) {
        throw std::runtime_error("parameter_key cannot be empty.");
    }
    check_gate_is_valid(gate);
    if (!gate.is_parametric()) {
        throw std::runtime_error(
            "Circuit::add_gate(const Gate&, double): non-parametric gate cannot add with "
            "parameter_key.");
    }
    _gate_list.push_back(gate->copy());
    _parameter_key_list.push_back(parameter_key);
}
void Circuit::add_gate(Gate&& gate, std::string_view parameter_key) {
    if (parameter_key.empty()) {
        throw std::runtime_error("parameter_key cannot be empty.");
    }
    check_gate_is_valid(gate);
    if (!gate.is_parametric()) {
        throw std::runtime_error(
            "Circuit::add_gate(Gate&&, double): non-parametric gate cannot add with "
            "parameter_key.");
    }
    _gate_list.push_back(std::move(gate));
    _parameter_key_list.push_back(parameter_key);
}
void Circuit::add_circuit(const Circuit& circuit) {
    if (circuit._n_qubits != _n_qubits) {
        throw std::runtime_error(
            "Circuit::add_circuit(const Circuit&): circuit with different qubit count cannot be "
            "merged.");
    }
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    _parameter_key_list.reserve(_parameter_key_list.size() + circuit._parameter_key_list.size());
    for (const auto& gate : circuit._gate_list) {
        _gate_list.push_back(gate);
    }
    for (std::string_view parameter_key : circuit._parameter_key_list) {
        _parameter_key_list.push_back(parameter_key);
    }
}
void Circuit::add_circuit(Circuit&& circuit) {
    for (auto&& gate : circuit._gate_list) {
        add_gate(std::move(gate));
    }
    if (circuit._n_qubits != _n_qubits) {
        throw std::runtime_error(
            "Circuit::add_circuit(Circuit&&): circuit with different qubit count cannot be "
            "merged.");
    }
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    _parameter_key_list.reserve(_parameter_key_list.size() + circuit._parameter_key_list.size());
    for (auto&& gate : circuit._gate_list) {
        _gate_list.push_back(std::move(gate));
    }
    for (std::string_view parameter_key : circuit._parameter_key_list) {
        _parameter_key_list.push_back(parameter_key);
    }
}

void Circuit::update_quantum_state(StateVector& state) const {
    for (std::string_view parameter_key : _parameter_key_list) {
        if (parameter_key.empty()) continue;
        using namespace std::string_literals;
        throw std::runtime_error(
            "Circuit::update_quantum_state(StateVector&) const: parameter named "s +
            std::string(parameter_key) + "is not given.");
    }
    for (const auto& gate : _gate_list) {
        gate->update_quantum_state(state);
    }
}

void Circuit::update_quantum_state(StateVector& state,
                                   const std::map<std::string_view, double>& parameters) const {
    for (std::string_view parameter_key : _parameter_key_list) {
        if (parameter_key.empty()) continue;
        if (!parameters.contains(parameter_key)) {
            using namespace std::string_literals;
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVector&, const std::map<std::string_view, double>&) const: parameter named "s +
                std::string(parameter_key) + "is not given.");
        }
    }
    for (UINT gate_idx : std::views::iota(0, _gate_list.size())) {
        Gate gate = _gate_list[gate_idx];
        std::string_view parameter_key = _parameter_key_list[gate_idx];
        if (parameter_key.empty()) {
            gate->update_quantum_state(state);
        } else {
            gate.to_parametric_gate()->update_quantum_state(state, parameters.at(parameter_key));
        }
    }
}

Circuit Circuit::copy() const {
    Circuit ccircuit(_n_qubits);
    ccircuit._gate_list.reserve(_gate_list.size());
    ccircuit._parameter_key_list.reserve(_parameter_key_list.size());
    for (auto&& gate : _gate_list) {
        ccircuit._gate_list.push_back(gate->copy());
    }
    for (std::string_view parameter_key : _parameter_key_list) {
        ccircuit._parameter_key_list.push_back(parameter_key);
    }
    return ccircuit;
}

Circuit Circuit::get_inverse() const {
    if (!_parameter_key_list.empty()) {
        throw std::runtime_error(
            "Circuit::get_inverse() const: Circuit with parameter is not supported");
    }
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
