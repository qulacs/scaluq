#include "noisesimulator.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace scaluq {

NoiseSimulator::NoiseSimulator(const Circuit& init_circuit)
    : _circuit(init_circuit.copy()), _initial_state(init_circuit.n_qubits()) {
    _initial_state.set_zero_state();
}

NoiseSimulator::NoiseSimulator(const Circuit& init_circuit, const StateVector& init_state)
    : _circuit(init_circuit.copy()), _initial_state(init_state) {
    if (init_circuit.n_qubits() != init_state.n_qubits()) {
        throw std::invalid_argument("Circuit and initial state qubit counts do not match");
    }
}

NoiseSimulator::Result::Result(std::vector<std::pair<StateVector, std::uint64_t>> result) {
    std::transform(result.begin(), result.end(), std::back_inserter(_result), [](const auto& p) {
        return std::make_pair(p.first, p.second);
    });
}

std::vector<std::uint64_t> NoiseSimulator::Result::sampling() const {
    std::vector<std::uint64_t> sampling_result;
    for (const auto& [state, count] : _result) {
        auto state_samples = state.sampling(count);
        sampling_result.insert(sampling_result.end(), state_samples.begin(), state_samples.end());
    }
    return sampling_result;
}

std::vector<std::uint64_t> NoiseSimulator::execute(const std::uint64_t execution_count) {
    auto result = execute_and_get_result(execution_count);
    return result->sampling();
}

std::unique_ptr<NoiseSimulator::Result> NoiseSimulator::execute_and_get_result(
    const std::uint64_t sample_count) {
    auto sampling_required = generate_sampling_request(sample_count);
    auto simulate_result = simulate(sampling_required);
    return std::make_unique<Result>(std::move(simulate_result));
}

std::vector<std::unique_ptr<NoiseSimulator::SamplingRequest>>
NoiseSimulator::generate_sampling_request(const std::uint64_t sample_count) {
    std::vector<std::vector<std::uint64_t>> selected_gate_pos(
        sample_count, std::vector<std::uint64_t>(_circuit.gate_list().size(), 0));

    const std::uint64_t gate_size = _circuit.gate_list().size();
    for (std::uint64_t i = 0; i < sample_count; ++i) {
        for (std::uint64_t j = 0; j < gate_size; ++j) {
            const auto& gate_variant = _circuit.gate_list()[j];
            if (std::holds_alternative<Gate>(gate_variant)) {
                const auto& gate = std::get<Gate>(gate_variant);
                selected_gate_pos[i][j] = randomly_select_which_gate_pos_to_apply(gate);
            } else {
                auto p = std::get<std::pair<ParamGate, std::string>>(gate_variant);
                const auto& gate = p.first;
                selected_gate_pos[i][j] = randomly_select_which_gate_pos_to_apply(gate);
            }
        }
    }

    std::sort(selected_gate_pos.begin(), selected_gate_pos.end());
    std::reverse(selected_gate_pos.begin(), selected_gate_pos.end());

    std::vector<std::unique_ptr<SamplingRequest>> required_sampling_requests;
    std::uint64_t current_sampling_count = 0;

    for (std::uint64_t i = 0; i < sample_count; ++i) {
        current_sampling_count++;
        if (i + 1 == sample_count || selected_gate_pos[i] != selected_gate_pos[i + 1]) {
            required_sampling_requests.push_back(
                std::make_unique<SamplingRequest>(selected_gate_pos[i], current_sampling_count));
            current_sampling_count = 0;
        }
    }

    return required_sampling_requests;
}

std::vector<std::pair<StateVector, std::uint64_t>> NoiseSimulator::simulate(
    const std::vector<std::unique_ptr<SamplingRequest>>& sampling_requests,
    const std::map<std::string, double>& parameters) {
    std::vector<std::pair<StateVector, std::uint64_t>> simulation_result;
    if (sampling_requests.empty()) {
        return simulation_result;
    }

    StateVector common_state(_initial_state.n_qubits());
    StateVector buffer(_initial_state.n_qubits());

    common_state = _initial_state;
    std::size_t done_itr = 0;

    for (const auto& request : sampling_requests) {
        const auto& current_gate_pos = request->gate_pos;

        while (done_itr < current_gate_pos.size() && current_gate_pos[done_itr] == 0) {
            const auto& gate_variant = _circuit.gate_list()[done_itr];

            if (std::holds_alternative<Gate>(gate_variant)) {
                const auto& gate = std::get<Gate>(gate_variant);
                if (!gate->is_noise()) {
                    gate->update_quantum_state(common_state);
                } else {
                    ProbablisticGate prob_gate(gate);
                    prob_gate->gate_list()[current_gate_pos[done_itr]]->update_quantum_state(
                        common_state);
                }
            } else {
                const auto& [param_gate, key] =
                    std::get<std::pair<ParamGate, std::string>>(gate_variant);
                if (!param_gate->is_noise()) {
                    param_gate->update_quantum_state(common_state, parameters.at(key));
                } else {
                    ParamProbablisticGate prob_gate(param_gate);
                    const auto& p_gate_variant = prob_gate->gate_list()[current_gate_pos[done_itr]];
                    if (std::holds_alternative<Gate>(p_gate_variant)) {
                        const auto& gate = std::get<Gate>(p_gate_variant);
                        gate->update_quantum_state(common_state);
                    } else {
                        const auto& gate = std::get<ParamGate>(p_gate_variant);
                        gate->update_quantum_state(common_state, parameters.at(key));
                    }
                }
            }
            done_itr++;
        }

        buffer = common_state;
        apply_gates(current_gate_pos, buffer, done_itr, parameters);
        simulation_result.emplace_back(buffer, request->num_of_sampling);
    }
    return simulation_result;
}

template <IsValidGate GateType>
std::uint64_t NoiseSimulator::randomly_select_which_gate_pos_to_apply(const GateType& gate) {
    if (!gate->is_noise()) {
        return 0;
    }

    using ProbGateType =
        std::conditional_t<std::is_same_v<GateType, Gate>, ProbablisticGate, ParamProbablisticGate>;

    ProbGateType prob_gate(gate);
    const auto& current_cumulative_distribution = prob_gate->get_cumulative_distribution();

    double tmp = _random.uniform();
    auto gate_iterator = std::lower_bound(
        current_cumulative_distribution.begin(), current_cumulative_distribution.end(), tmp);

    auto gate_pos = std::distance(current_cumulative_distribution.begin(), gate_iterator);
    return std::max((std::uint64_t)0, (std::uint64_t)gate_pos - (std::uint64_t)1);
}

void NoiseSimulator::apply_gates(const std::vector<std::uint64_t>& chosen_gate,
                                 StateVector& sampling_state,
                                 std::size_t start_pos,
                                 const std::map<std::string, double>& parameters) {
    const std::uint64_t gate_size = _circuit.gate_list().size();

    for (std::uint64_t q = start_pos; q < gate_size; ++q) {
        const auto& gate_variant = _circuit.gate_list()[q];

        if (std::holds_alternative<Gate>(gate_variant)) {
            const auto& gate = std::get<Gate>(gate_variant);
            if (!gate->is_noise()) {
                gate->update_quantum_state(sampling_state);
            } else {
                ProbablisticGate prob_gate(gate);
                prob_gate->gate_list()[chosen_gate[q]]->update_quantum_state(sampling_state);
            }
        } else {
            const auto& [param_gate, key] =
                std::get<std::pair<ParamGate, std::string>>(gate_variant);
            if (!param_gate->is_noise()) {
                param_gate->update_quantum_state(sampling_state, parameters.at(key));
            } else {
                ParamProbablisticGate prob_gate(param_gate);
                const auto& p_gate_variant = prob_gate->gate_list()[chosen_gate[q]];
                if (std::holds_alternative<Gate>(p_gate_variant)) {
                    const auto& gate = std::get<Gate>(p_gate_variant);
                    gate->update_quantum_state(sampling_state);
                } else {
                    const auto& gate = std::get<ParamGate>(p_gate_variant);
                    gate->update_quantum_state(sampling_state, parameters.at(key));
                }
            }
        }
    }
}

}  // namespace scaluq
