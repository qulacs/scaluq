#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_probablistic.hpp>
#include <scaluq/gate/param_gate_probablistic.hpp>

#include "../util/template.hpp"

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

    int a = 0;
    for (auto& g : _gate_list) {
        std::cout << "Depth: " << ++a << std::endl;
        std::vector<double> probs;
        std::vector<GateWithKey> gates;
        if (g.index() == 0) {
            const auto& gate = std::get<0>(g);
            if (gate.gate_type() == GateType::Probablistic) {
                probs = ProbablisticGate<Prec, Space>(gate)->distribution();
                const auto& gate_list = ProbablisticGate<Prec, Space>(gate)->gate_list();
                for (const auto& tmp : gate_list) {
                    gates.push_back(tmp);
                }
            } else {
                probs = std::vector<double>{1.0};
                gates.push_back(gate);
            }
        } else {
            const auto& [gate, key] = std::get<1>(g);
            if (gate.param_gate_type() == ParamGateType::ParamProbablistic) {
                probs = ParamProbablisticGate<Prec, Space>(gate)->distribution();
                auto prob_gate_list = ParamProbablisticGate<Prec, Space>(gate)->gate_list();
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
        for (std::uint64_t i = 0; i < states.batch_size(); ++i) {
            for (std::uint64_t _ = 0; _ < scounts[i]; ++_) {
                ++gate_used_count[i][dist(mt)];
            }
        }

        std::uint64_t new_size = 0;
        for (std::uint64_t i = 0; i < gate_used_count.size(); ++i) {
            for (std::uint64_t j = 0; j < gate_used_count[i].size(); ++j) {
                if (gate_used_count[i][j] == 0) continue;
                ++new_size;
            }
        }

        new_states = StateVectorBatched<Prec, Space>(new_size, initial_state.n_qubits());
        new_scounts.assign(new_size, 0);

        std::int64_t insert_idx = 0;
        for (std::uint64_t i = 0; i < probs.size(); ++i) {
            StateVectorBatched<Prec, Space> tmp_states = states.copy();
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
    if (gate.gate_type() == GateType::Probablistic) {
        for (auto g : ProbablisticGate<Prec, Space>(gate)->gate_list()) {
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
    if (gate.param_gate_type() == ParamGateType::ParamProbablistic) {
        for (auto g : ParamProbablisticGate<Prec, Space>(gate)->gate_list()) {
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

SCALUQ_DECLARE_CLASS_FOR_PRECISION_AND_EXECUTION_SPACE(Circuit)
}  // namespace scaluq
