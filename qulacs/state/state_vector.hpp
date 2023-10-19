#pragma once

#include <Kokkos_Core.hpp>
#include <vector>

#include "../types.hpp"

class StateVector {
    UINT _n_qubits;
    UINT _dim;
    Kokkos::View<Complex*> _amplitudes;

public:
    StateVector(UINT n_qubits);

    static StateVector Haar_random_state(UINT n_qubits);

    [[nodiscard]] UINT n_qubits() const;

    [[nodiscard]] UINT dim() const;

    [[nodiscard]] Kokkos::View<Complex*>& amplitudes_raw();
    [[nodiscard]] const Kokkos::View<Complex*>& amplitudes_raw() const;

    [[nodiscard]] const std::vector<Complex>& amplitudes() const;

    [[nodiscard]] Complex& operator[](const int index);
    [[nodiscard]] const Complex& operator[](const int index) const;

    [[nodiscard]] double compute_squared_norm() const;

    void normalize();
};
