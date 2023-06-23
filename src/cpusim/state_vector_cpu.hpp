#pragma once
#include <core/state_vector.hpp>
#include <vector>

class StateVectorCpu : StateVector {
    UINT _n_qubits;
    int _dim;
    std::vector<Complex> _amplitudes;

public:
    StateVectorCpu(UINT n_qubits);

    UINT n_qubits() const override;

    int dim() const override;

    Complex& operator[](const int index) override;

    const Complex& operator[](const int index) const override;
};
