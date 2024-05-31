#include "state_vector.hpp"

#include <bit>

#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace scaluq {

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _raw(StateVectorView(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim)) {
    set_zero_state();
}

void StateVector::set_amplitude_at_index(UINT index, const Complex& c) {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
}

Complex StateVector::get_amplitude_at_index(UINT index) const {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
    return host_view();
}

void StateVector::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at_index(0, 1);
}

void StateVector::set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }

void StateVector::set_computational_basis(UINT basis) {
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(UINT): "
            "index of "
            "computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at_index(basis, 1);
}

StateVector StateVector::Haar_random_state(UINT n_qubits, UINT seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVector state(n_qubits);
    Kokkos::parallel_for(
        state._dim, KOKKOS_LAMBDA(UINT i) {
            auto rand_gen = rand_pool.get_state();
            state._raw[i] = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    state.normalize();
    return state;
}

UINT StateVector::n_qubits() const { return this->_n_qubits; }

UINT StateVector::dim() const { return this->_dim; }

std::vector<Complex> StateVector::amplitudes() const {
    return internal::convert_device_view_to_host_vector(_raw);
}

double StateVector::get_squared_norm() const {
    double norm = 0.;
    Kokkos::parallel_reduce(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(UINT it, double& tmp) { tmp += squared_norm(this->_raw[it]); },
        norm);
    return norm;
}

void StateVector::normalize() {
    const auto norm = std::sqrt(this->get_squared_norm());
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(UINT it) { this->_raw[it] /= norm; });
}

double StateVector::get_zero_probability(UINT target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(UINT): index "
            "of target qubit must be smaller than qubit_count");
    }
    double sum = 0.;
    Kokkos::parallel_reduce(
        "zero_prob",
        _dim >> 1,
        KOKKOS_CLASS_LAMBDA(UINT i, double& lsum) {
            UINT basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += squared_norm(this->_raw[basis_0]);
        },
        sum);
    return sum;
}

double StateVector::get_marginal_probability(const std::vector<UINT>& measured_values) const {
    if (measured_values.size() != _n_qubits) {
        throw std::runtime_error(
            "Error: "
            "StateVector::get_marginal_probability(vector<UINT>): "
            "the length of measured_values must be equal to qubit_count");
    }

    std::vector<UINT> target_index;
    std::vector<UINT> target_value;
    for (UINT i = 0; i < measured_values.size(); ++i) {
        UINT measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        } else if (measured_value != StateVector::UNMEASURED) {
            throw std::runtime_error(
                "Error: Invalid qubit state specified. Each qubit state must be 0, 1, or "
                "StateVector::UNMEASURED.");
        }
    }

    double sum = 0.;
    auto d_target_index = internal::convert_host_vector_to_device_view(target_index);
    auto d_target_value = internal::convert_host_vector_to_device_view(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        _dim >> target_index.size(),
        KOKKOS_CLASS_LAMBDA(UINT i, double& lsum) {
            UINT basis = i;
            for (UINT cursor = 0; cursor < d_target_index.size(); cursor++) {
                UINT insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis ^= d_target_value[cursor] << insert_index;
            }
            lsum += squared_norm(this->_raw[basis]);
        },
        sum);

    return sum;
}

double StateVector::get_entropy() const {
    double ent = 0;
    const double eps = 1e-15;
    Kokkos::parallel_reduce(
        "get_entropy",
        _dim,
        KOKKOS_CLASS_LAMBDA(UINT idx, double& lsum) {
            double prob = squared_norm(_raw[idx]);
            prob = (prob > eps) ? prob : eps;
            lsum += -prob * Kokkos::log(prob);
        },
        ent);
    return ent;
}

void StateVector::add_state_vector(const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] += state._raw[i]; });
}

void StateVector::add_state_vector_with_coef(const Complex& coef, const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] += coef * state._raw[i]; });
}

void StateVector::multiply_coef(const Complex& coef) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] *= coef; });
}

std::vector<UINT> StateVector::sampling(UINT sampling_count, UINT seed) const {
    Kokkos::View<double*> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        _dim,
        KOKKOS_CLASS_LAMBDA(UINT i, double& update, const bool final) {
            double prob = squared_norm(this->_raw[i]);
            if (final) {
                stacked_prob[i + 1] = update + prob;
            }
            update += prob;
        });

    Kokkos::View<UINT*> result(Kokkos::ViewAllocateWithoutInitializing("result"), sampling_count);
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::parallel_for(
        sampling_count, KOKKOS_LAMBDA(UINT i) {
            auto rand_gen = rand_pool.get_state();
            double r = rand_gen.drand(0., 1.);
            UINT lo = 0, hi = stacked_prob.size();
            while (hi - lo > 1) {
                UINT mid = (lo + hi) / 2;
                if (stacked_prob[mid] > r) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            result[i] = lo;
            rand_pool.free_state(rand_gen);
        });
    return internal::convert_device_view_to_host_vector(result);
}

void StateVector::load(const std::vector<Complex>& other) {
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: StateVector::load(vector<Complex>&): invalid "
            "length of state");
    }
    _raw = internal::convert_host_vector_to_device_view(other);
}

void StateVector::load(StateVectorView raw) {
    const UINT dm = raw.size();
    if ((dm & (dm - 1)) > 0) {
        throw std::runtime_error(
            "Error: StateVector::StateVector(StateVectorView): the size of the view does "
            "not match as statevector.");
    }
    _n_qubits = (dm >> std::countr_zero(dm));
    _dim = dm;
    _raw = raw;
}

StateVector StateVector::copy() const {
    StateVector new_vec(_n_qubits);
    Kokkos::deep_copy(new_vec._raw, _raw);
    return new_vec;
}

std::ostream& operator<<(std::ostream& os, const StateVector& state) {
    os << state.to_string();
    return os;
}

}  // namespace scaluq
