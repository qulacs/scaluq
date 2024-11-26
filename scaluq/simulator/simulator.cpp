#include "simulator.hpp"

namespace scaluq {
CircuitSimulator::CircuitSimulator(const Circuit& cirq, const StateVector& initial_state = nullptr)
    : _cirq(std::make_shared<Circuit>(cirq)), _initial_state(initial_state) {
    if (_initial_state == nullptr) {
        _initial_state = StateVector(_cirq.n_qubits());
    }
}

void CircuitSimulator::initialize_state(std::uint64_t computational_basis = 0) {
    _initial_state.set_computational_basis(computational_basis);
}

void CircuitSimulator::initialize_random_state() {
    _initial_state = StateVector::Haar_random_state(_cirq.n_qubits());
}

void CircuitSimulator::initialize_random_state(std::uint64_t seed) {
    _initial_state = StateVector::Haar_random_state(_cirq.n_qubits(), seed);
}

void CircuitSimulator::simulate() { _cirq->update_quantum_state(_initial_state); }

void CircuitSimulator::simulate_range(std::uint64_t start_index, std::uint64_t end_index) {
    // not implemented yet
    // _cirq->update_quantum_state(_initial_state, start_index, end_index);
}

void CircuitSimulator::get_expectation_value(const Operator& op) {
    return op.get_expectation_value(_initial_state);
}

std::uint64_t CircuitSimulator::get_gate_count() const { return _cirq.n_gates(); }

void CircuitSimulator::copy_state_to_buffer() { _buffer = _initial_state; }

void CircuitSimulator::copy_state_from_buffer() { _initial_state = _buffer; }

void CircuitSimulator::swap_state_and_buffer() {
    if (_buffer == nullptr) {
        _buffer = new StateVector(_initial_state.n_qubits());
        _buffer.set_zero_state();
    }
    std::swap(_initial_state, _buffer);
}

}  // namespace scaluq
