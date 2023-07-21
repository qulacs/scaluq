#pragma once
#include <core/state_vector.hpp>
#include <vector>

class StateVectorCpu : StateVector {
    UINT _n_qubits;
    int _dim;
    std::vector<Complex> _amplitudes;

public:
    StateVectorCpu(UINT n_qubits);

    static StateVectorCpu Haar_random_state(UINT n_qubits);

    UINT n_qubits() const override;

    int dim() const override;

    const std::vector<Complex>& amplitudes() const override;

    Complex& operator[](const int index) override;

    const Complex& operator[](const int index) const override;

    double compute_squared_norm() const override;

    void normalize() override;
};
