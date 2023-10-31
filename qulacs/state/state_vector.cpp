#include "state_vector.hpp"

#include "../util/random.hpp"

namespace qulacs {
StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    this->_amplitudes[0] = 1.0 + 0.0i;
}

StateVector StateVector::Haar_random_state(UINT n_qubits) {
    StateVector state_vector(n_qubits);
    Random rng;
    for (int i = 0; i < state_vector.dim(); i++) {
        state_vector[i] = Complex(rng.normal(), rng.normal());
    }
    state_vector.normalize();
    return state_vector;
}

UINT StateVector::n_qubits() const { return this->_n_qubits; }

UINT StateVector::dim() const { return this->_dim; }

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }

const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }

Complex& StateVector::operator[](const int index) { return this->_amplitudes[index]; }

const Complex& StateVector::operator[](const int index) const { return this->_amplitudes[index]; }

double StateVector::compute_squared_norm() const {
    double norm = 0.;
    Kokkos::parallel_reduce(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& it, double& tmp) {
            tmp += std::norm(this->_amplitudes[it]);
        },
        norm);
    return norm;
}

void StateVector::normalize() {
    const auto norm = std::sqrt(this->compute_squared_norm());
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& it) { this->_amplitudes[it] /= norm; });
}
}  // namespace qulacs
