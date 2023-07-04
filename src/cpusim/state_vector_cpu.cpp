#include "state_vector_cpu.hpp"

#include <core/types.hpp>

StateVectorCpu::StateVectorCpu(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(std::vector<Complex>(this->_dim, 0.0 + 0.0i)) {
    this->_amplitudes[0] = 1.0 + 0.0i;
}

UINT StateVectorCpu::n_qubits() const { return this->_n_qubits; }

int StateVectorCpu::dim() const { return this->_dim; }

Complex& StateVectorCpu::operator[](const int index) { return this->_amplitudes[index]; }

const Complex& StateVectorCpu::operator[](const int index) const {
    return this->_amplitudes[index];
}
