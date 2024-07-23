#include "circuit.hpp"

#include <ranges>

#include "../gate/gate_factory.hpp"
#include "../gate/merge_gate.hpp"

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
void Circuit::add_param_gate(const ParamGate& param_gate, std::string_view parameter_key) {
    check_gate_is_valid(param_gate);
    _gate_list.push_back(std::make_pair(param_gate->copy(), std::string(parameter_key)));
}
void Circuit::add_param_gate(ParamGate&& param_gate, std::string_view parameter_key) {
    check_gate_is_valid(param_gate);
    _gate_list.push_back(std::make_pair(std::move(param_gate), std::string(parameter_key)));
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
                                   const std::map<std::string, double>& parameters) const {
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

Circuit Circuit::copy() const {
    Circuit ccircuit(_n_qubits);
    ccircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0)
            ccircuit._gate_list.push_back(std::get<0>(gate)->copy());
        else {
            const auto& [param_gate, key] = std::get<1>(gate);
            ccircuit._gate_list.push_back(std::make_pair(param_gate->copy(), key));
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
            const auto& [param_gate, key] = std::get<1>(gate);
            icircuit._gate_list.push_back(std::make_pair(param_gate->get_inverse(), key));
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

void Circuit::check_gate_is_valid(const ParamGate& param_gate) const {
    auto targets = param_gate->get_target_qubit_list();
    auto controls = param_gate->get_control_qubit_list();
    bool valid = true;
    if (!targets.empty()) valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
    if (!controls.empty()) valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
    if (!valid) {
        throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
    }
}

void Circuit::optimize(UINT block_size) {
    if (block_size >= 3) {
        throw std::runtime_error(
            "Currently block_size >= 3 is not supported because general matrix gate with qubits >= "
            "3 is not implemented.");
    }
    std::vector<GateWithKey> new_gate_list;
    double global_phase = 0.;
    std::vector<std::pair<Gate, std::vector<UINT>>> gate_pool;
    constexpr UINT NO_GATES = std::numeric_limits<UINT>::max();
    std::vector<UINT> latest_gate_idx(_n_qubits, NO_GATES);
    for (const GateWithKey& gate_with_key : _gate_list) {
        if (gate_with_key.index() == 1) {
            const auto& pgate = std::get<1>(gate_with_key).first;
            for (UINT target : pgate->get_target_qubit_list()) {
                latest_gate_idx[target] = NO_GATES;
            }
            for (UINT control : pgate->get_control_qubit_list()) {
                latest_gate_idx[control] = NO_GATES;
            }
            new_gate_list.emplace_back(std::move(gate_with_key));
            continue;
        }
        const auto& gate = std::get<0>(gate_with_key);
        if (gate.gate_type() == GateType::I) {
            continue;
        }
        if (gate.gate_type() == GateType::GlobalPhase) {
            global_phase += GlobalPhaseGate(gate)->phase();
            continue;
        }
        auto target_list = gate->get_target_qubit_list();
        auto control_list = gate->get_control_qubit_list();
        std::vector<UINT> targets;
        targets.reserve(target_list.size() + control_list.size());
        std::ranges::copy(target_list, std::back_inserter(targets));
        std::ranges::copy(control_list, std::back_inserter(targets));
        std::vector<UINT> previous_gate_indices;
        std::vector<UINT> newly_applied_qubits;
        for (UINT target : targets) {
            if (latest_gate_idx[target] == NO_GATES) {
                newly_applied_qubits.push_back(target);
            } else {
                previous_gate_indices.push_back(latest_gate_idx[target]);
            }
        }
        previous_gate_indices.erase(std::ranges::unique(previous_gate_indices).begin(),
                                    previous_gate_indices.end());
        UINT merged_gate_size =
            std::accumulate(previous_gate_indices.begin(),
                            previous_gate_indices.end(),
                            newly_applied_qubits.size(),
                            [&](UINT sz, UINT idx) { return sz + gate_pool[idx].second.size(); });
        auto is_pauli = [](const Gate& gate) {
            GateType type = gate.gate_type();
            return type == GateType::I || type == GateType::X || type == GateType::Y ||
                   type == GateType::Z || type == GateType::Pauli;
        };
        bool all_pauli =
            is_pauli(gate) && std::ranges::all_of(previous_gate_indices, [&](UINT idx) {
                return is_pauli(gate_pool[idx].first);
            });
        if (!all_pauli && merged_gate_size > block_size) {
            for (UINT idx : previous_gate_indices) {
                for (UINT qubit : gate_pool[idx].second) {
                    latest_gate_idx[qubit] = NO_GATES;
                }
                new_gate_list.emplace_back(std::move(gate_pool[idx].first));
            }
            UINT new_idx = gate_pool.size();
            for (UINT qubit : targets) {
                latest_gate_idx[qubit] = new_idx;
            }
            gate_pool.emplace_back(std::move(gate), std::move(targets));
            continue;
        }
        Gate merged_gate = gate::I();
        UINT new_idx = gate_pool.size();
        std::vector<UINT> new_targets;
        for (UINT idx : previous_gate_indices) {
            double phase;
            std::tie(merged_gate, phase) = merge_gate(merged_gate, gate_pool[idx].first);
            global_phase += phase;
            for (UINT qubit : gate_pool[idx].second) {
                new_targets.push_back(qubit);
                latest_gate_idx[qubit] = new_idx;
            }
        }
        {
            double phase;
            std::tie(merged_gate, phase) = merge_gate(merged_gate, gate);
            global_phase += phase;
            for (UINT qubit : newly_applied_qubits) {
                new_targets.push_back(qubit);
                latest_gate_idx[qubit] = new_idx;
            }
        }
        gate_pool.emplace_back(std::move(merged_gate), std::move(new_targets));
    }
    std::ranges::sort(latest_gate_idx);
    latest_gate_idx.erase(std::ranges::unique(latest_gate_idx).begin(), latest_gate_idx.end());
    for (UINT idx : latest_gate_idx) {
        if (idx == NO_GATES) continue;
        new_gate_list.emplace_back(std::move(gate_pool[idx].first));
    }
    if (std::abs(global_phase) < 1e-12) new_gate_list.push_back(gate::GlobalPhase(global_phase));
    _gate_list.swap(new_gate_list);
}
}  // namespace scaluq
