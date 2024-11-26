#pragma once

#include <memory>
#include <optional>

#include "../circuit/circuit.hpp"
#include "../gate/gate_factory.hpp"
#include "../gate/merge_gate.hpp"
#include "../operator/operator.hpp"
#include "../state/state_vector.hpp"
#include "../state/state_vector_batched.hpp"

namespace scaluq {
class CircuitSimulator {
private:
    std::shared_ptr<Circuit> _cirq;
    StateVector _initial_state, _buffer;

public:
    /*
     * constructor
     * @param circuit: Circuit to simulate
     * @param initial_state: Initial state of the circuit
     */
    CircuitSimulator(const Circuit& circuit, const StateVector& initial_state = nullptr);

    /*
     * initialize quantum state to computational basis
     */
    void initialize_state(std::uint64_t computational_basis = 0);

    /*
     * initialize quantum state to random state
     */
    void initialize_random_state();
    void initialize_random_state(std::uint64_t seed);

    /*
     * simulate quantum circuit
     */
    void simulate();

    /*
     * simulate quantum circuit from start-index to end-index
     * @param start_index: start index of the circuit
     * @param end_index: end index of the circuit
     */
    void simulate_range(std::uint64_t start_index, std::uint64_t end_index);

    /*
     * compute expectation value of operator
     * @param operator: Operator to compute expectation value
     * @return: Expectation value of the operator
     */
    Complex get_expectation_value(const Operator& op);

    /*
     * get number of gates in the circuit
     */
    std::uint64_t get_gate_count() const;

    /*
     * copy quantum state to buffer
     */
    void copy_state_to_buffer();

    /*
     * copy buffer to quantum state
     */
    void copy_state_from_buffer();

    /*
     * swap quantum state and buffer
     */
    void swap_state_and_buffer();

    /*
     * get quantum state
     */
    const StateVector& get_state() const { return _initial_state; }
};
}  // namespace scaluq
