#include "state_vector.hpp"

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    this->_amplitudes[0] = 1.0 + 0.0i;
}

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }
const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }
