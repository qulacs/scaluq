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

    UINT n_qubits() const;

    UINT dim() const;

    Kokkos::View<Complex*>& amplitudes_raw();
    const Kokkos::View<Complex*>& amplitudes_raw() const;

    const std::vector<Complex>& amplitudes() const;

    Complex& operator[](const int index);
    const Complex& operator[](const int index) const;

    double compute_squared_norm() const;

    void normalize();
};
