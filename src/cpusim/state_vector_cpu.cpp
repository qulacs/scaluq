#include "state_vector_cpu.hpp"

#include <core/random.hpp>
#include <core/types.hpp>

StateVectorCpu::StateVectorCpu(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(std::vector<Complex>(this->_dim, 0.0 + 0.0i)) {
    this->_amplitudes[0] = 1.0 + 0.0i;
}

StateVectorCpu StateVectorCpu::Haar_random_state(UINT n_qubits) {
    StateVectorCpu state_vector(n_qubits);
    Random rng;
    for (int i = 0; i < state_vector.dim(); i++) {
        state_vector[i] = Complex(rng.uniform(), rng.uniform());
    }
    state_vector.normalize();
    return state_vector;
}

UINT StateVectorCpu::n_qubits() const { return this->_n_qubits; }

int StateVectorCpu::dim() const { return this->_dim; }

const std::vector<Complex>& StateVectorCpu::amplitudes() const { return this->_amplitudes; }

Complex& StateVectorCpu::operator[](const int index) { return this->_amplitudes[index]; }

const Complex& StateVectorCpu::operator[](const int index) const {
    return this->_amplitudes[index];
}

double StateVectorCpu::compute_squared_norm() const {
    double norm = 0.;
    for (const auto& amplitude : this->_amplitudes) {
        norm += std::norm(amplitude);
    }
    return norm;
}

void StateVectorCpu::normalize() {
    const auto norm = std::sqrt(this->compute_squared_norm());
    for (auto& amplitude : this->_amplitudes) {
        amplitude /= norm;
    }
}
