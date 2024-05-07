#include "circuit.hpp"

#include <ranges>

namespace scaluq {
UINT Circuit::calculate_depth() const {
    std::vector<UINT> filled_step(_n_qubits, 0ULL);
    for (const auto& gate : _gate_list) {
        std::vector<UINT> control_qubits = gate.index() == 0
                                               ? std::get<0>(gate)->get_control_qubit_list()
                                               : std::get<1>(gate).first->get_control_qubit_list();
        std::vector<UINT> target_qubits = gate.index() == 0
                                              ? std::get<0>(gate)->get_target_qubit_list()
                                              : std::get<1>(gate).first->get_target_qubit_list();
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
void Circuit::add_pgate(const PGate& pgate, std::string_view parameter_key) {
    check_gate_is_valid(pgate);
    _gate_list.push_back(std::make_pair(pgate->copy(), parameter_key));
}
void Circuit::add_pgate(PGate&& pgate, std::string_view parameter_key) {
    check_gate_is_valid(pgate);
    _gate_list.push_back(std::make_pair(std::move(pgate), parameter_key));
}
void Circuit::add_circuit(const Circuit& circuit) {
    if (circuit._n_qubits != _n_qubits) {
        throw std::runtime_error(
            "Circuit::add_circuit(const Circuit&): circuit with different qubit count cannot be "
            "merged.");
    }
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    for (const auto& gate : circuit._gate_list) {
        _gate_list.push_back(gate);
    }
}
void Circuit::add_circuit(Circuit&& circuit) {
    if (circuit._n_qubits != _n_qubits) {
        throw std::runtime_error(
            "Circuit::add_circuit(Circuit&&): circuit with different qubit count cannot be "
            "merged.");
    }
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    for (auto&& gate : circuit._gate_list) {
        _gate_list.push_back(std::move(gate));
    }
}

void Circuit::update_quantum_state(StateVector& state,
                                   const std::map<std::string_view, double>& parameters) const {
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) continue;
        std::string_view key = std::get<1>(gate).second;
        if (!parameters.contains(key)) {
            using namespace std::string_literals;
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVector&, const std::map<std::string_view, double>&) const: parameter named "s +
                std::string(key) + "is not given.");
        }
    }
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(state);
        } else {
            const auto& [pgate, key] = std::get<1>(gate);
            pgate->update_quantum_state(state, parameters.at(key));
        }
    }
}

Circuit Circuit::copy() const {
    Circuit ccircuit(_n_qubits);
    ccircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0)
            ccircuit._gate_list.push_back(std::get<0>(gate)->copy());
        else {
            const auto& [pgate, key] = std::get<1>(gate);
            ccircuit._gate_list.push_back(std::make_pair(pgate->copy(), key));
        }
    }
    return ccircuit;
}

Circuit Circuit::get_inverse() const {
    Circuit icircuit(_n_qubits);
    icircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list | std::views::reverse) {
        if (gate.index() == 0)
            icircuit._gate_list.push_back(std::get<0>(gate)->get_inverse());
        else {
            const auto& [pgate, key] = std::get<1>(gate);
            icircuit._gate_list.push_back(std::make_pair(pgate->get_inverse(), key));
        }
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

void Circuit::check_gate_is_valid(const PGate& pgate) const {
    auto targets = pgate->get_target_qubit_list();
    auto controls = pgate->get_control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}
}  // namespace scaluq
