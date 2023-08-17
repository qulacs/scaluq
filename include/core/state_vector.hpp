#pragma once

#include <vector>

#include "types.hpp"

class StateVector {
public:
    virtual UINT n_qubits() const = 0;

    virtual int dim() const = 0;

    virtual const std::vector<Complex>& amplitudes() const = 0;

    virtual Complex& operator[](const int index) = 0;

    virtual const Complex& operator[](const int index) const = 0;

    virtual double compute_squared_norm() const = 0;

    virtual void normalize() = 0;
};
