#pragma once
#include <Kokkos_Core.hpp>
#include <core/state_vector.hpp>
#include <vector>

class StateVectorKokkos : StateVector {
    UINT _n_qubits;
    int _dim;
    Kokkos::View<Complex*> _amplitudes;

public:
    StateVectorKokkos(UINT n_qubits);

    static StateVectorKokkos Haar_random_state(UINT n_qubits);

    UINT n_qubits() const override;

    int dim() const override;

    const std::vector<Complex>& amplitudes() const override;

    Complex& operator[](const int index) override;

    const Complex& operator[](const int index) const override;

    double compute_squared_norm() const override;

    void normalize() override;
};
