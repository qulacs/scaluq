#include "state_vector.hpp"
#include "../util/random.hpp"

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    this->_amplitudes[0] = 1.0 + 0.0i;
}

static StateVector Haar_random_state(UINT n_qubits) const {
    StateVector state_vector(n_qubits);
    Random rng;
    for (int i = 0; i < state_vector.dim(); i++) {
        state_vector[i] = Complex(rng.normal(), rng.normal());
    }
    state_vector.normalize();
    return state_vector;
}

UINT n_qubits() const { return this->_n_qubits; }

UINT dim() const { return this->_dim; }

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }

const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }

const std::vector<Complex>& amplitudes() const { return this->_amplitudes; }

Complex& StateVector::operator[](const int index) & { return this->_amplitudes[index]; }

const Complex& StateVector::operator[](const int index) const& {
    return this->_amplitudes[index];
}

double StateVector::compute_squared_norm() const {
    double norm = 0.;
    for (const auto& amplitude : this->_amplitudes) {
        norm += std::norm(amplitude);
    }
    return norm;
}

void StateVector::normalize() {
    const auto norm = std::sqrt(this->compute_squared_norm());
    for (auto& amplitude : this->_amplitudes) {
        amplitude /= norm;
    }
}
