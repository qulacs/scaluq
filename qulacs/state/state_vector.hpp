#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <stdexcept>
#include <vector>

#include "../types.hpp"

namespace qulacs {
class StateVector {
    UINT _n_qubits;
    UINT _dim;

public:
    Kokkos::View<Complex*> _raw;
    StateVector() = default;
    StateVector(UINT n_qubits);
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at_index(const UINT& index, const Complex& c);

    /**
     * @attention Very slow. You should use amplitudes() instead if you can.
     */
    [[nodiscard]] Complex get_amplitude_at_index(const UINT& index) const;

    [[nodiscard]] static StateVector Haar_random_state(UINT n_qubits, UINT seed);
    [[nodiscard]] static StateVector Haar_random_state(UINT n_qubits);

    /**
     * @brief zero-fill
     */
    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(UINT basis);

    [[nodiscard]] UINT n_qubits() const;

    [[nodiscard]] UINT dim() const;

    [[nodiscard]] std::vector<Complex> amplitudes() const;

    [[nodiscard]] double compute_squared_norm() const;

    void normalize();

    [[nodiscard]] double get_zero_probability(UINT target_qubit_index) const;
    [[nodiscard]] double get_marginal_probability(const std::vector<UINT>& measured_values) const;
    [[nodiscard]] double get_entropy() const;

    void add_state_vector(const StateVector& state);
    void add_state_vector_with_coef(const Complex& coef, const StateVector& state);
    void multiply_coef(const Complex& coef);

    [[nodiscard]] std::vector<UINT> sampling(UINT sampling_count, UINT seed = 0) const;

    [[nodiscard]] std::string to_string() const;

    void load(const std::vector<Complex>& other);
    [[nodiscard]] StateVector copy() const;

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state);
};
}  // namespace qulacs
