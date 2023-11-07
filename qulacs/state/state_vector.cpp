#include "state_vector.hpp"

#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace qulacs {

/**
 * Insert 0 to qubit_index-th bit of basis_index. basis_mask must be 1ULL <<
 * qubit_index.
 */
inline static UINT insert_zero_to_basis_index(UINT basis_index, UINT basis_mask, UINT qubit_index) {
    UINT temp_basis = (basis_index >> qubit_index) << (qubit_index + 1);
    return temp_basis + basis_index % basis_mask;
}

double marginal_prob(const std::vector<UINT>& sorted_target_qubit_index_list,
                     const std::vector<UINT>& measured_value_list,
                     UINT target_qubit_index_count,
                     const Kokkos::View<Complex*>& state,
                     UINT dim) {
    UINT loop_dim = dim >> target_qubit_index_count;
    double sum = 0.;

    // Create views on the device
    auto d_sorted_target_qubit_index_list =
        convert_host_vector_to_device_view(sorted_target_qubit_index_list);
    auto d_measured_value_list = convert_host_vector_to_device_view(measured_value_list);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<>(0, loop_dim),
        KOKKOS_LAMBDA(const UINT& i, double& lsum) {
            UINT basis = i;
            for (UINT cursor = 0; cursor < target_qubit_index_count; cursor++) {
                UINT insert_index = d_sorted_target_qubit_index_list[cursor];
                UINT mask = 1ULL << insert_index;
                basis = insert_zero_to_basis_index(basis, mask, insert_index);
                basis ^= mask * d_measured_value_list[cursor];
            }
            lsum += std::norm(state[basis]);
        },
        sum);

    return sum;
}

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

void StateVector::set_zero_norm_state() { Kokkos::deep_copy(_amplitudes, 0); }

void StateVector::set_computational_basis(UINT basis) {
    Kokkos::deep_copy(_amplitudes, 0);
    assert(basis < _dim);  // TODO ちゃんと例外を投げる
    _amplitudes[basis] = 1;
}

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

double StateVector::get_zero_probability(UINT target_qubit_index) const {
    assert(target_qubit_index < _dim);  // TODO ちゃんと例外を投げる
    const UINT loop_dim = _dim / 2;
    const UINT mask = 1ULL << target_qubit_index;
    double sum = 0.;
    Kokkos::parallel_reduce(
        loop_dim,
        KOKKOS_CLASS_LAMBDA(const int i, double& lsum) {
            UINT basis_0 = insert_zero_to_basis_index(i, mask, target_qubit_index);
            lsum += std::norm(this->_amplitudes[basis_0]);
        },
        sum);
    return sum;
}

double StateVector::get_marginal_probability(const std::vector<UINT>& measured_values) const {
    assert(measured_values.size() == this->_n_qubits);

    std::vector<UINT> target_index;
    std::vector<UINT> target_value;
    for (UINT i = 0; i < measured_values.size(); ++i) {
        UINT measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        }
    }
    return marginal_prob(target_index, target_value, (UINT)target_index.size(), _amplitudes, _dim);
}

void StateVector::add_state_with_coef(const Complex& coef, const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] += coef * state[i]; });
}

}  // namespace qulacs
