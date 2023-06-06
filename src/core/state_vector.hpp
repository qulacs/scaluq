#pragma once

#include "type.hpp"

class StateVector {
public:
    virtual UINT n_qubits() const = 0;

    virtual int dim() const = 0;

    virtual Complex& operator[](const int index) = 0;
};
