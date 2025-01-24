#include <scaluq/circuit/circuit.hpp>

#include "../util/template.hpp"

namespace scaluq {
FLOAT_AND_SPACE(Fp, Sp)
std::set<std::string> Circuit<Fp, Sp>::key_set() const {
    std::set<std::string> key_set;
    for (auto&& gate : _gate_list) {
        if (gate.index() == 1) key_set.insert(std::get<1>(gate).second);
    }
    return key_set;
}

FLOAT_AND_SPACE(Fp, Sp)
std::uint64_t Circuit<Fp, Sp>::calculate_depth() const {
    std::vector<std::uint64_t> filled_step(_n_qubits, 0ULL);
    for (const auto& gate : _gate_list) {
        std::vector<std::uint64_t> control_qubits =
            gate.index() == 0 ? std::get<0>(gate)->control_qubit_list()
                              : std::get<1>(gate).first->control_qubit_list();
        std::vector<std::uint64_t> target_qubits =
            gate.index() == 0 ? std::get<0>(gate)->target_qubit_list()
                              : std::get<1>(gate).first->target_qubit_list();
        std::uint64_t max_step_amount_target_qubits = 0;
        for (std::uint64_t control : control_qubits) {
            if (max_step_amount_target_qubits < filled_step[control]) {
                max_step_amount_target_qubits = filled_step[control];
            }
        }
        for (std::uint64_t target : control_qubits) {
            if (max_step_amount_target_qubits < filled_step[target]) {
                max_step_amount_target_qubits = filled_step[target];
            }
        }
        for (std::uint64_t control : control_qubits) {
            filled_step[control] = max_step_amount_target_qubits + 1;
        }
        for (std::uint64_t target : target_qubits) {
            filled_step[target] = max_step_amount_target_qubits + 1;
        }
    }
    return *std::ranges::max_element(filled_step);
}

FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::add_circuit(const Circuit<Fp, Sp>& circuit) {
    if (circuit._n_qubits != _n_qubits) {
        throw std::runtime_error(
            "Circuit::add_circuit(const Circuit&): circuit with different qubit count cannot "
            "be merged.");
    }
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    for (const auto& gate : circuit._gate_list) {
        _gate_list.push_back(gate);
    }
}
FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::add_circuit(Circuit<Fp, Sp>&& circuit) {
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

FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::update_quantum_state(StateVector<Fp, Sp>& state,
                                           const std::map<std::string, Fp>& parameters) const {
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) continue;
        const auto& key = std::get<1>(gate).second;
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
            const auto& [param_gate, key] = std::get<1>(gate);
            param_gate->update_quantum_state(state, parameters.at(key));
        }
    }
}

FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::update_quantum_state(
    StateVectorBatched<Fp, Sp>& states,
    const std::map<std::string, std::vector<Fp>>& parameters) const {
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) continue;
        const auto& key = std::get<1>(gate).second;
        if (!parameters.contains(key)) {
            using namespace std::string_literals;
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVector&, const std::map<std::string_view, double>&) const: parameter named "s +
                std::string(key) + "is not given.");
        }
    }
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) {
            std::get<0>(gate)->update_quantum_state(states);
        } else {
            const auto& [param_gate, key] = std::get<1>(gate);
            param_gate->update_quantum_state(states, parameters.at(key));
        }
    }
}

FLOAT_AND_SPACE(Fp, Sp)
Circuit<Fp, Sp> Circuit<Fp, Sp>::copy() const {
    Circuit ccircuit(_n_qubits);
    ccircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0)
            ccircuit._gate_list.push_back(std::get<0>(gate));
        else {
            const auto& [param_gate, key] = std::get<1>(gate);
            ccircuit._gate_list.push_back(std::make_pair(param_gate, key));
        }
    }
    return ccircuit;
}

FLOAT_AND_SPACE(Fp, Sp)
Circuit<Fp, Sp> Circuit<Fp, Sp>::get_inverse() const {
    Circuit icircuit(_n_qubits);
    icircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list | std::views::reverse) {
        if (gate.index() == 0)
            icircuit._gate_list.push_back(std::get<0>(gate)->get_inverse());
        else {
            const auto& [param_gate, key] = std::get<1>(gate);
            icircuit._gate_list.push_back(std::make_pair(param_gate->get_inverse(), key));
        }
    }
    return icircuit;
}

FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::check_gate_is_valid(const Gate<Fp, Sp>& gate) const {
    auto targets = gate->target_qubit_list();
    auto controls = gate->control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}

FLOAT_AND_SPACE(Fp, Sp)
void Circuit<Fp, Sp>::check_gate_is_valid(const ParamGate<Fp, Sp>& gate) const {
    auto targets = gate->target_qubit_list();
    auto controls = gate->control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}

FLOAT_AND_SPACE_DECLARE_CLASS(Circuit)
}  // namespace scaluq
