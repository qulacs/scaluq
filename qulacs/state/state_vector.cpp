#include "state_vector.hpp"

#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace qulacs {

/**
 * Insert 0 to insert_index-th bit of basis_index.
 */
KOKKOS_INLINE_FUNCTION UINT insert_zero_to_basis_index(UINT basis_index, UINT insert_index) {
    UINT mask = (1ULL << insert_index) - 1;
    UINT temp_basis = (basis_index >> insert_index) << (insert_index + 1);
    return temp_basis | (basis_index & mask);
}

inline void write_to_device_at_index(StateVector& v, const int index, const Complex& c) {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(v.amplitudes_raw(), index), host_view());
}

inline Complex read_from_device_at_index(StateVector& v, const int index) {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(v.amplitudes_raw(), index));
    return host_view();
}

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    set_zero_state();
}

void StateVector::set_zero_state() {
    Kokkos::deep_copy(_amplitudes, 0);
    write_to_device_at_index(*this, 0, 1);
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
    write_to_device_at_index(*this, basis, 1);
}

StateVector StateVector::Haar_random_state(UINT n_qubits, UINT seed) {
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

StateVector StateVector::Haar_random_state(UINT n_qubits) {
    std::random_device rd;
    Kokkos::Random_XorShift64_Pool<> rand_pool(rd());
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

UINT StateVector::n_qubits() const { return this->_n_qubits; }

UINT StateVector::dim() const { return this->_dim; }

Kokkos::View<Complex*>& StateVector::amplitudes_raw() { return this->_amplitudes; }

const Kokkos::View<Complex*>& StateVector::amplitudes_raw() const { return this->_amplitudes; }

std::vector<Complex> StateVector::amplitudes() const {
    return convert_device_view_to_host_vector(_amplitudes);
}

Complex& StateVector::operator[](const int index) { return this->_amplitudes[index]; }

const Complex& StateVector::operator[](const int index) const { return this->_amplitudes[index]; }

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
        KOKKOS_CLASS_LAMBDA(const int i, double& lsum) {
            UINT basis_0 = insert_zero_to_basis_index(i, target_qubit_index);
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
                basis = insert_zero_to_basis_index(basis, insert_index);
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
        KOKKOS_CLASS_LAMBDA(const int& i, double& update, const bool final) {
            double prob = norm2(this->_amplitudes[i]);
            if (final) {
                stacked_prob[i + 1] = update + prob;
            }
            update += prob;
        });

    Kokkos::View<UINT*> result("result", sampling_count);
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::parallel_for(
        sampling_count, KOKKOS_CLASS_LAMBDA(const UINT& i) {
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
    os << " *** Quantum State ***" << std::endl;
    os << " * Qubit Count : " << _n_qubits << std::endl;
    os << " * Dimension   : " << _dim << std::endl;
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

std::ostream& operator<<(std::ostream& os, const StateVector& state) {
    os << state.to_string();
    return os;
}

}  // namespace qulacs
