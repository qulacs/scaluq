#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <vector>

#include "../types.hpp"

namespace qulacs {
class StateVector {
    UINT _n_qubits;
    UINT _dim;
    Kokkos::View<Complex*> _amplitudes;

public:
    StateVector(UINT n_qubits);

    static StateVector Haar_random_state(UINT n_qubits, uint64_t seed);
    static StateVector Haar_random_state(UINT n_qubits);

    void set_zero_state();
    void set_zero_norm_state();
    void set_computational_basis(UINT basis);

    UINT n_qubits() const;

    UINT dim() const;

    Kokkos::View<Complex*>& amplitudes_raw();
    const Kokkos::View<Complex*>& amplitudes_raw() const;

    std::vector<Complex> amplitudes() const;

    Complex& operator[](const int index);
    const Complex& operator[](const int index) const;

    double compute_squared_norm() const;

    void normalize();
};
}  // namespace qulacs
