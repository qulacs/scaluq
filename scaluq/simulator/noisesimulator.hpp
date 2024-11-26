#pragma once

#include <memory>
#include <optional>

#include "../circuit/circuit.hpp"
#include "../gate/gate_factory.hpp"
#include "../gate/merge_gate.hpp"
#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"

namespace scaluq {

class NoiseSimulator {
private:
    Random _random;
    Circuit _circuit;
    StateVector _initial_state;

    class SamplingRequest {
    public:
        std::vector<std::uint64_t> gate_pos;
        std::uint64_t num_of_sampling;

        SamplingRequest(std::vector<std::uint64_t> init_gate_pos,
                        std::uint64_t init_num_of_sampling)
            : gate_pos(std::move(init_gate_pos)), num_of_sampling(init_num_of_sampling) {}
    };

    void apply_gates(const std::vector<std::uint64_t>& chosen_gate,
                     StateVector& sampling_state,
                     std::size_t start_pos);

    std::vector<std::unique_ptr<SamplingRequest>> generate_sampling_request(
        std::uint64_t sample_count);

    std::uint64_t randomly_select_which_gate_pos_to_apply(const Gate& gate);
    std::uint64_t randomly_select_which_gate_pos_to_apply(const ParamGate& gate);

    std::vector<std::pair<StateVector, std::uint64_t>> simulate(
        const std::vector<std::unique_ptr<SamplingRequest>>& sampling_request_vector);

public:
    class Result {
    private:
        std::vector<std::pair<StateVector, std::uint64_t>> _result;

    public:
        explicit Result(std::vector<std::pair<StateVector, std::uint64_t>> result);
        std::vector<std::uint64_t> sampling() const;

        ~Result() = default;
    };

    explicit NoiseSimulator(const Circuit& init_circuit);
    NoiseSimulator(const Circuit& init_circuit, const StateVector& init_state);

    virtual ~NoiseSimulator() = default;

    virtual std::vector<std::uint64_t> execute(std::uint64_t sample_count);
    virtual std::unique_ptr<Result> execute_and_get_result(std::uint64_t execution_count);
};

}  // namespace scaluq
