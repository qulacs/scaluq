#include "state_vector.hpp"

#include "../util/random.hpp"

namespace qulacs {
StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    set_zero_state();
}

void StateVector::set_zero_state() {
    Kokkos::deep_copy(_amplitudes, 0);
    _amplitudes[0] = 1;
}

/**
 * @brief zero-fill
 */
void StateVector::set_zero_norm_state() { Kokkos::deep_copy(_amplitudes, 0); }

void StateVector::set_computational_basis(UINT basis) {
    Kokkos::deep_copy(_amplitudes, 0);
    assert(basis < _dim);  // TODO ちゃんと例外を投げる
    _amplitudes[basis] = 1;
}

/**
 * @param seed The seed value for the random number generator. If omitted, 0 is used.
 */
StateVector StateVector::Haar_random_state(UINT n_qubits, uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVector state(n_qubits);
    Kokkos::parallel_for(
        state._dim, KOKKOS_LAMBDA(const int i) {
            auto rand_gen = rand_pool.get_state();
            state._amplitudes[i] = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    state.normalize();
    return state;
}

StateVector StateVector::Haar_random_state(UINT n_qubits) { return Haar_random_state(n_qubits, 0); }

UINT StateVector::n_qubits() const { return this->_n_qubits; }

UINT StateVector::dim() const { return this->_dim; }

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }

const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }

std::vector<Complex> StateVector::amplitudes() const {
    std::vector<Complex> host_vector(_dim);
    auto host_mirror_view = Kokkos::create_mirror_view(_amplitudes);
    Kokkos::deep_copy(host_mirror_view, _amplitudes);
    std::copy(host_mirror_view.data(), host_mirror_view.data() + _dim, host_vector.begin());
    return host_vector;
}

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
