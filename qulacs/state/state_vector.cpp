#include "state_vector.hpp"

#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace qulacs {

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    set_zero_state();
}

void StateVector::set_amplitude_at_index(const UINT& index, const Complex& c) {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_amplitudes, index), host_view());
}

Complex StateVector::get_amplitude_at_index(const UINT& index) const {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_amplitudes, index));
    return host_view();
}

void StateVector::set_zero_state() {
    Kokkos::deep_copy(_amplitudes, 0);
    set_amplitude_at_index(0, 1);
}

void StateVector::set_zero_norm_state() { Kokkos::deep_copy(_amplitudes, 0); }

void StateVector::set_computational_basis(UINT basis) {
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(UINT): "
            "index of "
            "computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::deep_copy(_amplitudes, 0);
    set_amplitude_at_index(basis, 1);
}

StateVector StateVector::Haar_random_state(UINT n_qubits, UINT seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVector state(n_qubits);
    auto amp = state.amplitudes_raw();
    Kokkos::parallel_for(
        state._dim, KOKKOS_LAMBDA(const UINT& i) {
            auto rand_gen = rand_pool.get_state();
            amp[i] = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    state.normalize();
    return state;
}

StateVector StateVector::Haar_random_state(UINT n_qubits) {
    std::random_device rd;
    Kokkos::Random_XorShift64_Pool<> rand_pool(rd());
    StateVector state(n_qubits);
    auto amp = state.amplitudes_raw();
    Kokkos::parallel_for(
        state._dim, KOKKOS_LAMBDA(const UINT& i) {
            auto rand_gen = rand_pool.get_state();
            amp[i] = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    state.normalize();
    return state;
}

UINT StateVector::n_qubits() const { return this->_n_qubits; }

UINT StateVector::dim() const { return this->_dim; }

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }

const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }

std::vector<Complex> StateVector::amplitudes() const {
    return convert_device_view_to_host_vector(_amplitudes);
}

double StateVector::compute_squared_norm() const {
    double norm = 0.;
    Kokkos::parallel_reduce(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& it, double& tmp) { tmp += norm2(this->_amplitudes[it]); },
        norm);
    return norm;
}

void StateVector::normalize() {
    const auto norm = std::sqrt(this->compute_squared_norm());
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& it) { this->_amplitudes[it] /= norm; });
}

double StateVector::get_zero_probability(UINT target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(UINT): index "
            "of target qubit must be smaller than qubit_count");
    }
    const UINT loop_dim = _dim / 2;
    double sum = 0.;
    Kokkos::parallel_reduce(
        "zero_prob",
        loop_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& i, double& lsum) {
            UINT basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += norm2(this->_amplitudes[basis_0]);
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
        }
    }

    UINT loop_dim = _dim >> target_index.size();
    double sum = 0.;

    auto d_target_index = convert_host_vector_to_device_view(target_index);
    auto d_target_value = convert_host_vector_to_device_view(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        loop_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& i, double& lsum) {
            UINT basis = i;
            for (UINT cursor = 0; cursor < d_target_index.size(); cursor++) {
                UINT insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis ^= d_target_value[cursor] << insert_index;
            }
            lsum += norm2(this->_amplitudes[basis]);
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
        KOKKOS_CLASS_LAMBDA(const UINT& idx, double& lsum) {
            double prob = norm2(_amplitudes[idx]);
            prob = (prob > eps) ? prob : eps;
            lsum += -prob * Kokkos::log(prob);
        },
        ent);
    return ent;
}

void StateVector::add_state_vector(const StateVector& state) {
    auto amp = state.amplitudes_raw();
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] += amp[i]; });
}

void StateVector::add_state_vector_with_coef(const Complex& coef, const StateVector& state) {
    auto amp = state.amplitudes_raw();
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] += coef * amp[i]; });
}

void StateVector::multiply_coef(const Complex& coef) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] *= coef; });
}

std::vector<UINT> StateVector::sampling(UINT sampling_count, UINT seed) const {
    Kokkos::View<double*> stacked_prob("prob", _dim + 1);
    Kokkos::deep_copy(stacked_prob, 0);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        _dim,
        KOKKOS_CLASS_LAMBDA(const UINT& i, double& update, const bool final) {
            double prob = norm2(this->_amplitudes[i]);
            if (final) {
                stacked_prob[i + 1] = update + prob;
            }
            update += prob;
        });

    Kokkos::View<UINT*> result("result", sampling_count);
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::parallel_for(
        sampling_count, KOKKOS_LAMBDA(const UINT& i) {
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
    return convert_device_view_to_host_vector<UINT>(result);
}

std::string StateVector::to_string() const {
    std::stringstream os;
    auto amp = this->amplitudes();
    os << " *** Quantum State ***\n";
    os << " * Qubit Count : " << _n_qubits << '\n';
    os << " * Dimension   : " << _dim << '\n';
    os << " * State vector : \n";
    for (UINT i = 0; i < _dim; ++i) {
        os << amp[i] << std::endl;
    }
    return os.str();
}

void StateVector::load(const std::vector<Complex>& other) {
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: QuantumStateCpu::load(vector<Complex>&): invalid "
            "length of state");
    }
    _amplitudes = convert_host_vector_to_device_view(other);
}

StateVector StateVector::copy() const {
    StateVector new_vec(_n_qubits);
    auto new_amp = new_vec.amplitudes_raw();
    Kokkos::parallel_for(
        _dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { new_amp[i] = _amplitudes[i]; });
    return new_vec;
}

std::ostream& operator<<(std::ostream& os, const StateVector& state) {
    os << state.to_string();
    return os;
}

}  // namespace qulacs
