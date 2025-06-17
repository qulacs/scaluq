#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../prec_space.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
std::set<std::string> Circuit<Prec, Space>::key_set() const {
    std::set<std::string> key_set;
    for (auto&& gate : _gate_list) {
        if (gate.index() == 1) key_set.insert(std::get<1>(gate).second);
    }
    return key_set;
}

template <Precision Prec, ExecutionSpace Space>
std::uint64_t Circuit<Prec, Space>::calculate_depth() const {
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

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::add_circuit(const Circuit<Prec, Space>& circuit) {
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
template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::add_circuit(Circuit<Prec, Space>&& circuit) {
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

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::update_quantum_state(
    StateVector<Prec, Space>& state, const std::map<std::string, double>& parameters) const {
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

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states,
    const std::map<std::string, std::vector<double>>& parameters) const {
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

template <Precision Prec, ExecutionSpace Space>
Circuit<Prec, Space> Circuit<Prec, Space>::copy() const {
    Circuit<Prec, Space> ccircuit(_n_qubits);
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

template <Precision Prec, ExecutionSpace Space>
Circuit<Prec, Space> Circuit<Prec, Space>::get_inverse() const {
    Circuit<Prec, Space> icircuit(_n_qubits);
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

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::optimize(std::uint64_t max_block_size) {
    std::vector<GateWithKey> new_gate_list;  // result
    double global_phase = 0.;
    std::vector<Gate<Prec, Space>> gate_pool;  // waitlist for merge or push
    constexpr std::uint64_t NO_GATES = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> waiting_gate_idx_at_qubit(
        _n_qubits,
        NO_GATES);  // which gate is waiting in the qubit (by index of gate_pool)
    auto get_operand_list = [&](const GateWithKey& gate_with_key) {
        if (gate_with_key.index() == 0)
            return std::get<0>(gate_with_key)->operand_qubit_list();
        else
            return std::get<1>(gate_with_key).first->operand_qubit_list();
    };
    auto push = [&](const GateWithKey& gate_with_key) {
        // clear waitlist and push gate to result
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            waiting_gate_idx_at_qubit[operand] = NO_GATES;
        }
        new_gate_list.push_back(gate_with_key);
    };
    auto wait = [&](const Gate<Prec, Space>& gate) {
        // wait for merge or push
        std::uint64_t new_idx = gate_pool.size();
        for (std::uint64_t operand : gate->operand_qubit_list()) {
            waiting_gate_idx_at_qubit[operand] = new_idx;
        }
        gate_pool.push_back(gate);
    };
    auto get_waiting_gate_indices = [&](const GateWithKey& gate_with_key) {
        std::vector<std::uint64_t> waiting_gate_indices;
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            if (waiting_gate_idx_at_qubit[operand] != NO_GATES)
                waiting_gate_indices.push_back(waiting_gate_idx_at_qubit[operand]);
        }
        std::ranges::sort(waiting_gate_indices);
        waiting_gate_indices.erase(std::ranges::unique(waiting_gate_indices).begin(),
                                   waiting_gate_indices.end());
        return waiting_gate_indices;
    };
    auto push_waiting_gates = [&](const GateWithKey& gate_with_key) {
        for (std::uint64_t idx : get_waiting_gate_indices(gate_with_key)) {
            push(gate_pool[idx]);
        }
    };
    auto get_newly_applied_qubits = [&](const GateWithKey& gate_with_key) {
        std::vector<std::uint64_t> newly_applied_qubits;
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            if (waiting_gate_idx_at_qubit[operand] == NO_GATES)
                newly_applied_qubits.push_back(operand);
        }
        return newly_applied_qubits;
    };

    for (const GateWithKey& gate_with_key : _gate_list) {
        if (gate_with_key.index() == 1 ||
            std::get<0>(gate_with_key).gate_type() == GateType::Probabilistic) {
            // ParamGate and Probabilistic cannot be merged with others
            push_waiting_gates(gate_with_key);
            push(gate_with_key);
            continue;
        }
        const Gate<Prec, Space>& gate = std::get<0>(gate_with_key);
        if (gate.gate_type() == GateType::I) continue;  // IGate is ignored
        if (gate.gate_type() == GateType::GlobalPhase && gate->control_qubit_mask() == 0ULL) {
            // non-controlled GlobalPhase is ignored
            global_phase += GlobalPhaseGate<Prec, Space>(gate)->phase();
            continue;
        }
        std::vector<std::uint64_t> waiting_gate_indices = get_waiting_gate_indices(gate_with_key);
        if (waiting_gate_indices.empty()) {
            // just wait
            wait(gate);
            continue;
        }
        std::vector<std::uint64_t> newly_applied_qubits = get_newly_applied_qubits(gate_with_key);
        std::uint64_t new_gate_size =
            std::accumulate(waiting_gate_indices.begin(),
                            waiting_gate_indices.end(),
                            newly_applied_qubits.size(),
                            [&](std::uint64_t acc, std::uint64_t idx) {
                                return acc + gate_pool[idx]->operand_qubit_list().size();
                            });
        auto is_pauli = [&](const Gate<Prec, Space>& gate) {
            return gate.gate_type() == GateType::X || gate.gate_type() == GateType::Y ||
                   gate.gate_type() == GateType::Z || gate.gate_type() == GateType::Pauli;
        };
        auto is_pure_pauli = [&](const Gate<Prec, Space>& gate) {
            return gate->control_qubit_mask() == 0ULL && is_pauli(gate);
        };
        bool all_pauli = is_pure_pauli(gate) &&
                         std::ranges::all_of(waiting_gate_indices, [&](std::uint64_t idx) {
                             return is_pure_pauli(gate_pool[idx]);
                         });
        if (waiting_gate_indices.size() == 1) {
            const Gate<Prec, Space>& previous_gate = gate_pool[waiting_gate_indices[0]];
            // common control qubits are not counted as size
            std::uint64_t control_qubit_mask1 = gate->control_qubit_mask();
            std::uint64_t control_value_mask1 = gate->control_value_mask();
            std::uint64_t control_qubit_mask2 = previous_gate->control_qubit_mask();
            std::uint64_t control_value_mask2 = previous_gate->control_value_mask();
            new_gate_size -= std::popcount(control_qubit_mask1 & control_qubit_mask2 &
                                           ~(control_value_mask1 ^ control_value_mask2));

            // check whether both gates are pauli with same control qubits
            if (control_qubit_mask1 == control_qubit_mask2 &&
                control_value_mask1 == control_value_mask2 && is_pauli(gate) &&
                is_pauli(previous_gate)) {
                all_pauli = true;
            }
        }
        if (all_pauli || new_gate_size <= max_block_size) {
            // merge with waiting gates
            Gate<Prec, Space> merged_gate = gate;
            for (std::uint64_t idx : waiting_gate_indices) {
                const auto& [new_merged_gate, phase] =
                    merge_gate<Prec, Space>(gate_pool[idx], merged_gate);
                merged_gate = new_merged_gate;
                global_phase += phase;
            }

            // wait
            wait(merged_gate);
        } else {
            // not merge
            // push waiting gates
            for (std::uint64_t idx : waiting_gate_indices) {
                push(gate_pool[idx]);
            }

            // wait
            wait(gate);
        }
    }
    std::vector<std::uint64_t> finally_waiting_gate_indices;
    for (std::uint64_t idx : waiting_gate_idx_at_qubit) {
        if (idx != NO_GATES) finally_waiting_gate_indices.push_back(idx);
    }
    std::ranges::sort(finally_waiting_gate_indices);
    finally_waiting_gate_indices.erase(std::ranges::unique(finally_waiting_gate_indices).begin(),
                                       finally_waiting_gate_indices.end());
    for (std::uint64_t idx : finally_waiting_gate_indices) {
        new_gate_list.push_back(gate_pool[idx]);
    }
    if (std::abs(global_phase) > 1e-12)
        new_gate_list.push_back(gate::GlobalPhase<Prec, Space>(global_phase));
    _gate_list.swap(new_gate_list);
}

template <Precision Prec, ExecutionSpace Space>
std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> Circuit<Prec, Space>::simulate_noise(
    const StateVector<Prec, Space>& initial_state,
    std::uint64_t sampling_count,
    const std::map<std::string, double>& parameters,
    std::uint64_t seed) const {
    // サンプリング回数について，下図のような木をBFSする
    //    1000　　　　 {X:p=0.1, I:p=0.9}
    //   X/ ＼I
    //  100   900     {Z:p=0.2, I:p=0.8}
    // Z/＼I  Z/＼I
    // 20 80  180 720
    std::mt19937 mt(seed);
    StateVectorBatched<Prec, Space> states(1, initial_state.n_qubits()), new_states;
    states.set_state_vector_at(0, initial_state);
    // 今／次の深さにおける各状態のサンプリング回数
    std::vector<std::uint64_t> scounts{sampling_count}, new_scounts;

    for (auto& g : _gate_list) {
        std::vector<double> probs;
        std::vector<GateWithKey> gates;
        if (g.index() == 0) {
            const auto& gate = std::get<0>(g);
            if (gate.gate_type() == GateType::Probabilistic) {
                probs = ProbabilisticGate<Prec, Space>(gate)->distribution();
                const auto& gate_list = ProbabilisticGate<Prec, Space>(gate)->gate_list();
                for (const auto& tmp : gate_list) {
                    gates.push_back(tmp);
                }
            } else {
                probs = std::vector<double>{1.0};
                gates.push_back(gate);
            }
        } else {
            const auto& [gate, key] = std::get<1>(g);
            if (gate.param_gate_type() == ParamGateType::ParamProbabilistic) {
                probs = ParamProbabilisticGate<Prec, Space>(gate)->distribution();
                auto prob_gate_list = ParamProbabilisticGate<Prec, Space>(gate)->gate_list();
                for (const auto& tmp : prob_gate_list) {
                    if (tmp.index() == 0) {
                        gates.push_back(std::get<0>(tmp));
                    } else {
                        gates.push_back(std::pair<scaluq::ParamGate<Prec, Space>, std::string>{
                            std::get<1>(tmp), key});
                    }
                }
            } else {
                probs = std::vector<double>{1.0};
                gates.push_back(g);
            }
        }

        std::discrete_distribution<std::uint64_t> dist(probs.begin(), probs.end());
        // gate_used_count[i][j] := states[i] が gates[j] によって変換される回数
        std::vector<std::vector<std::uint64_t>> gate_used_count(
            states.batch_size(), std::vector<std::uint64_t>(probs.size(), 0));
        std::uint64_t new_size = 0;
        for (std::uint64_t i = 0; i < states.batch_size(); ++i) {
            for (std::uint64_t _ = 0; _ < scounts[i]; ++_) {
                std::uint64_t j = dist(mt);
                if (j >= probs.size()) {
                    throw std::runtime_error(
                        "Circuit::simulate_noise: discrete_distribution returned out of range "
                        "index.");
                }
                if (gate_used_count[i][j] == 0) {
                    ++new_size;
                }
                ++gate_used_count[i][j];
            }
        }

        new_states = StateVectorBatched<Prec, Space>::uninitialized_state(new_size,
                                                                          initial_state.n_qubits());
        new_scounts.assign(new_size, 0);

        std::int64_t insert_idx = 0;
        for (std::uint64_t i = 0; i < probs.size(); ++i) {
            StateVectorBatched<Prec, Space> tmp_states(states.copy());
            if (gates[i].index() == 0) {
                std::get<0>(gates[i])->update_quantum_state(tmp_states);
            } else {
                const auto& [param_gate, key] = std::get<1>(gates[i]);
                param_gate->update_quantum_state(
                    tmp_states, std::vector<double>(tmp_states.batch_size(), parameters.at(key)));
            }
            for (std::uint64_t j = 0; j < tmp_states.batch_size(); ++j) {
                if (gate_used_count[j][i] == 0) continue;
                new_states.set_state_vector_at(insert_idx, tmp_states.get_state_vector_at(j));
                new_scounts[insert_idx] = gate_used_count[j][i];
                ++insert_idx;
            }
        }
        states = new_states;
        scounts.swap(new_scounts);
    }
    std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> result;
    result.reserve(states.batch_size());
    for (std::uint64_t i = 0; i < states.batch_size(); ++i) {
        result.emplace_back(states.get_state_vector_at(i), scounts[i]);
    }
    return result;
}

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::check_gate_is_valid(const Gate<Prec, Space>& gate) const {
    if (gate.gate_type() == GateType::Probabilistic) {
        for (auto g : ProbabilisticGate<Prec, Space>(gate)->gate_list()) {
            check_gate_is_valid(g);
        }
    } else {
        auto targets = gate->target_qubit_list();
        auto controls = gate->control_qubit_list();
        bool valid = true;
        if (!targets.empty() && *std::ranges::max_element(targets) >= _n_qubits) valid = false;
        if (!controls.empty() && *std::ranges::max_element(controls) >= _n_qubits) valid = false;
        if (!valid) {
            throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
void Circuit<Prec, Space>::check_gate_is_valid(const ParamGate<Prec, Space>& gate) const {
    if (gate.param_gate_type() == ParamGateType::ParamProbabilistic) {
        for (auto g : ParamProbabilisticGate<Prec, Space>(gate)->gate_list()) {
            if (g.index() == 0) {
                check_gate_is_valid(std::get<0>(g));
            } else {
                check_gate_is_valid(std::get<1>(g));
            }
        }
    } else {
        auto targets = gate->target_qubit_list();
        auto controls = gate->control_qubit_list();
        bool valid = true;
        if (!targets.empty())
            valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
        if (!controls.empty())
            valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
        if (!valid) {
            throw std::runtime_error("Gate to be added to the circuit has invalid qubit range");
        }
    }
}

template class Circuit<internal::Prec, internal::Space>;
}  // namespace scaluq
