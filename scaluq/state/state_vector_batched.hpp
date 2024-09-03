#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {
class StateVectorBatched {
    std::uint64_t _batch_size;
    std::uint64_t _n_qubits;
    std::uint64_t _dim;

public:
    StateVectorBatchedView _raw;
    StateVectorBatched() = default;
    StateVectorBatched(std::uint64_t batch_size, std::uint64_t n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::uint64_t batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector& state);
    void set_state_vector(std::uint64_t batch_id, const StateVector& state);
    [[nodiscard]] StateVector get_state_vector_at(std::uint64_t batch_id) const;

    void set_zero_state();
    void set_computational_basis(std::uint64_t basis);
    void set_zero_norm_state();

    [[nodiscard]] std::vector<std::vector<std::uint64_t>> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    [[nodiscard]] static StateVectorBatched Haar_random_states(
        std::uint64_t batch_size,
        std::uint64_t n_qubits,
        bool set_same_state,
        std::uint64_t seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<Complex>> get_amplitudes() const;

    [[nodiscard]] std::vector<double> get_squared_norm() const;

    void normalize();

    [[nodiscard]] std::vector<double> get_zero_probability(std::uint64_t target_qubit_index) const;
    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] std::vector<double> get_entropy() const;

    void add_state_vector(const StateVectorBatched& states);
    void add_state_vector_with_coef(const Complex& coef, const StateVectorBatched& states);
    void multiply_coef(const Complex& coef);

    void load(const std::vector<std::vector<Complex>>& states);
    [[nodiscard]] StateVectorBatched copy() const;

    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states);
};
}  // namespace scaluq
