#pragma once

#include "types.hpp"

class StateVector {
public:
    virtual UINT n_qubits() const = 0;

    virtual int dim() const = 0;

    virtual Complex& operator[](const int index) = 0;

    virtual const Complex& operator[](const int index) const = 0;
};
