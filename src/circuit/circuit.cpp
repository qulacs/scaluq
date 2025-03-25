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
    // 下図のような木を構築する
    //    1000
    //    / ＼
    //  100   900
    //  /＼    /＼
    // 10 90  90 810
    std::mt19937 mt(seed);
    std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> states, new_states;
    states.emplace_back(initial_state, sampling_count);
    for (auto&& g : _gate_list) {
        std::vector<double> probs(1, 1.0);
        if (g.index() == 0) {
            if (std::get<0>(g).gate_type() == GateType::Probablistic) {
                probs = ProbablisticGate<Prec, Space>(std::get<0>(g))->distribution();
            }
        } else {
            if (std::get<1>(g).first.param_gate_type() == ParamGateType::ParamProbablistic) {
                probs = ParamProbablisticGate<Prec, Space>(std::get<1>(g).first)->distribution();
            }
        }

        for (auto& [state, cnt] : states) {
            // 多項分布に基づいて，それぞれのゲートが何回選ばれるかを計算
            std::vector<std::uint64_t> counts(probs.size(), 0);
            std::discrete_distribution<std::uint64_t> dist(probs.begin(), probs.end());
            for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0, cnt)) {
                ++counts[dist(mt)];
            }

            StateVectorBatched<Prec, Space> states_before_update(probs.size(), state.n_qubits());
            states_before_update.set_state_vector(state);
            for (std::uint64_t i = 0; i < counts.size(); ++i) {
                if (counts[i] == 0) continue;
                auto tmp = states_before_update.get_state_vector_at(i);
                if (g.index() == 0) {  // NonProbablisticGate
                    std::get<0>(g)->update_quantum_state(tmp);
                } else {  // ProbablisticGate
                    const auto& key = std::get<1>(g).second;
                    auto either_gate =
                        ParamProbablisticGate<Prec, Space>(std::get<1>(g).first)->gate_list()[i];
                    if (either_gate.index() == 0) {  // Gate
                        std::get<0>(either_gate)->update_quantum_state(tmp);
                    } else {  // ParamGate
                        std::get<1>(either_gate)->update_quantum_state(tmp, parameters.at(key));
                    }
                }
                new_states.emplace_back(tmp, counts[i]);
            }
        }
        states.swap(new_states);
        new_states.clear();
    }
    return states;
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
        if (!targets.empty())
            valid &= *std::max_element(targets.begin(), targets.end()) < _n_qubits;
        if (!controls.empty())
            valid &= *std::max_element(controls.begin(), controls.end()) < _n_qubits;
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
