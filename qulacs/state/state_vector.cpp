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

StateVector::StateVector(UINT n_qubits)
    : _n_qubits(n_qubits),
      _dim(1 << n_qubits),
      _amplitudes(Kokkos::View<Complex*>("state", this->_dim)) {
    set_zero_state();
}

StateVector& StateVector::operator=(const StateVector& other) {
    _n_qubits = other.n_qubits();
    _dim = other.dim();
    Kokkos::deep_copy(_amplitudes, other.amplitudes_raw());
    return *this;
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
        "zero_prob",
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

    UINT loop_dim = _dim >> target_index.size();
    double sum = 0.;

    // Create views on the device
    auto d_sorted_target_qubit_index_list = convert_host_vector_to_device_view(target_index);
    auto d_measured_value_list = convert_host_vector_to_device_view(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<>(0, loop_dim),
        KOKKOS_CLASS_LAMBDA(const UINT& i, double& lsum) {
            UINT basis = i;
            for (UINT cursor = 0; cursor < d_sorted_target_qubit_index_list.size(); cursor++) {
                UINT insert_index = d_sorted_target_qubit_index_list[cursor];
                UINT mask = 1ULL << insert_index;
                basis = insert_zero_to_basis_index(basis, mask, insert_index);
                basis ^= mask * d_measured_value_list[cursor];
            }
            lsum += std::norm(this->_amplitudes[basis]);
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
            double prob = std::norm(_amplitudes[idx]);
            prob = (prob > eps) ? prob : eps;
            lsum += -prob * std::log(prob);
        },
        ent);
    return ent;
}

void StateVector::add_state(const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] += state[i]; });
}

void StateVector::add_state_with_coef(const Complex& coef, const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] += coef * state[i]; });
}

void StateVector::multiply_coef(const Complex& coef) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] *= coef; });
}

void StateVector::multiply_elementwise_function(const std::function<Complex(UINT)>& func) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(const UINT& i) { this->_amplitudes[i] *= func(i); });
}

/**
 * 未テスト！
 */
std::vector<UINT> StateVector::sampling(UINT sampling_count, UINT seed) const {
    Kokkos::vector<double> stacked_prob(_dim + 1, 0);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        _dim,
        KOKKOS_CLASS_LAMBDA(const int& i, double& update, const bool final) {
            double prob = std::norm(_amplitudes[i]);
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
            result[i] = stacked_prob.lower_bound(0, _dim + 1, rand_gen.drand(0., 1.)) - 1;
            rand_pool.free_state(rand_gen);
        });

    return convert_device_view_to_host_vector<UINT>(result);
}

std::string StateVector::to_string() const {
    std::stringstream os;
    os << " *** Quantum State ***" << std::endl;
    os << " * Qubit Count : " << _n_qubits << std::endl;
    os << " * Dimension   : " << _dim << std::endl;
    os << " * State vector : \n";
    for (UINT i = 0; i < _dim; ++i) {
        os << _amplitudes[i] << std::endl;
    }
    return os.str();
}

std::ostream& operator<<(std::ostream& os, const StateVector& state) {
    os << state.to_string();
    return os;
}

std::string StateVector::get_device_name() const {
#ifdef KOKKOS_ENABLE_CUDA
    return "gpu";
#endif
    return "cpu";
}

}  // namespace qulacs
