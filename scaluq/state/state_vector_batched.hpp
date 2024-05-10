#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "../types.hpp"

namespace scaluq {
class StateVectorBatched {
    UINT _batch_size;
    UINT _n_qubits;
    UINT _dim;

public:
    Kokkos::View<Complex*> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(UINT batch_size, UINT n_qubits);
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    UINT n_qubits() const { return this->_n_qubits; }

    UINT dim() const { return this->_dim; }

    UINT batch_size() const { return this->_batch_size; }

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at_index(const UINT index, const Complex& c);
    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at_index(const UINT batch_id, const UINT index, const Complex& c);

    /**
     * @attention Very slow. You should use amplitudes() instead if you can.
     */
    [[nodiscard]] std::vector<Complex> get_amplitude_at_index(const UINT index) const;
    /**
     * @attention Very slow. You should use amplitudes() instead if you can.
     */
    [[nodiscard]] Complex get_amplitude_at_index(const UINT batch_id, const UINT index) const;

    void set_zero_state();

    [[nodiscard]] static StateVectorBatched Haar_random_state(UINT batch_size,
                                                              UINT n_qubits,
                                                              UINT seed = std::random_device()());

    [[nodiscard]] std::vector<std::vector<Complex>> amplitudes() const;
    [[nodiscard]] std::vector<Complex> amplitudes(UINT batch_id) const;

    std::vector<double> get_squared_norm() const;

    void normalize();
};
}  // namespace scaluq
