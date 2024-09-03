#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "../types.hpp"

namespace scaluq {
class StateVector {
    std::uint64_t _n_qubits;
    std::uint64_t _dim;

public:
    static constexpr std::uint64_t UNMEASURED = 2;
    Kokkos::View<Complex*> _raw;
    StateVector() = default;
    StateVector(std::uint64_t n_qubits);
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at(std::uint64_t index, const Complex& c);

    /**
     * @attention Very slow. You should use get_amplitudes() instead if you can.
     */
    [[nodiscard]] Complex get_amplitude_at(std::uint64_t index) const;

    [[nodiscard]] static StateVector Haar_random_state(std::uint64_t n_qubits,
                                                       std::uint64_t seed = std::random_device()());

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(std::uint64_t basis);

    [[nodiscard]] std::uint64_t n_qubits() const;

    [[nodiscard]] std::uint64_t dim() const;

    [[nodiscard]] std::vector<Complex> get_amplitudes() const;

    [[nodiscard]] double get_squared_norm() const;

    void normalize();

    [[nodiscard]] double get_zero_probability(std::uint64_t target_qubit_index) const;
    [[nodiscard]] double get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const;
    [[nodiscard]] double get_entropy() const;

    void add_state_vector(const StateVector& state);
    void add_state_vector_with_coef(const Complex& coef, const StateVector& state);
    void multiply_coef(const Complex& coef);

    [[nodiscard]] std::vector<std::uint64_t> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const;

    void load(const std::vector<Complex>& other);

    [[nodiscard]] StateVector copy() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state);

    [[nodiscard]] std::string to_string() const;
};
}  // namespace scaluq
