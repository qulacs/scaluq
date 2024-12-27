#include <scaluq/state/state_vector.hpp>

#include "../util/math.hpp"
#include "../util/template.hpp"

namespace scaluq {
FLOAT(Fp)
StateVector<Fp>::StateVector(std::uint64_t n_qubits)
    : _n_qubits(n_qubits),
      _dim(1ULL << n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim) {
    set_zero_state();
}
FLOAT(Fp)
StateVector<Fp>::StateVector(Kokkos::View<ComplexType*> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}
FLOAT(Fp)
void StateVector<Fp>::set_amplitude_at(std::uint64_t index, ComplexType c) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
}
FLOAT(Fp)
Complex<Fp> StateVector<Fp>::get_amplitude_at(std::uint64_t index) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
    return host_view();
}
FLOAT(Fp)
StateVector<Fp> StateVector<Fp>::Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVector<Fp> state(n_qubits);
    Kokkos::parallel_for(
        state._dim, KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            state._raw(i) = ComplexType(static_cast<Fp>(rand_gen.normal(0.0, 1.0)),
                                        static_cast<Fp>(rand_gen.normal(0.0, 1.0)));
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    state.normalize();
    return state;
}
FLOAT(Fp)
void StateVector<Fp>::set_zero_state() {
    Kokkos::deep_copy(_raw, Fp{0});
    set_amplitude_at(0, Fp{1});
}
FLOAT(Fp)
void StateVector<Fp>::set_zero_norm_state() { Kokkos::deep_copy(_raw, {0}); }
FLOAT(Fp)
void StateVector<Fp>::set_computational_basis(std::uint64_t basis) {
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(std::uint64_t): "
            "index of computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::deep_copy(_raw, Fp{0});
    set_amplitude_at(basis, Fp{1});
}
FLOAT(Fp)
void StateVector<Fp>::set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    *this = Haar_random_state(n_qubits, seed);
}
FLOAT(Fp)
[[nodiscard]] std::vector<typename StateVector<Fp>::ComplexType> StateVector<Fp>::get_amplitudes()
    const {
    return internal::convert_device_view_to_host_vector<ComplexType>(_raw);
}
FLOAT(Fp)
Fp StateVector<Fp>::get_squared_norm() const {
    Fp norm;
    Kokkos::parallel_reduce(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t it, Fp & tmp) {
            tmp += internal::squared_norm(this->_raw[it]);
        },
        internal::Sum<Fp, Kokkos::DefaultExecutionSpace>(norm));
    return norm;
}
FLOAT(Fp)
void StateVector<Fp>::normalize() {
    const Fp norm = internal::sqrt(this->get_squared_norm());
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(std::uint64_t it) { this->_raw[it] /= norm; });
    Kokkos::fence();
}
FLOAT(Fp)
Fp StateVector<Fp>::get_zero_probability(std::uint64_t target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(std::uint64_t): index "
            "of target qubit must be smaller than qubit_count");
    }
    Fp sum = 0;
    Kokkos::parallel_reduce(
        "zero_prob",
        _dim >> 1,
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += internal::squared_norm(this->_raw[basis_0]);
        },
        internal::Sum<Fp, Kokkos::DefaultExecutionSpace>(sum));
    return sum;
}
FLOAT(Fp)
Fp StateVector<Fp>::get_marginal_probability(
    const std::vector<std::uint64_t>& measured_values) const {
    if (measured_values.size() != _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_marginal_probability(const vector<std::uint64_t>&): "
            "the length of measured_values must be equal to qubit_count");
    }

    std::vector<std::uint64_t> target_index;
    std::vector<std::uint64_t> target_value;
    for (std::uint64_t i = 0; i < measured_values.size(); ++i) {
        std::uint64_t measured_value = measured_values[i];
        if (measured_value == 0 || measured_value == 1) {
            target_index.push_back(i);
            target_value.push_back(measured_value);
        } else if (measured_value != StateVector::UNMEASURED) {
            throw std::runtime_error(
                "Error: StateVector::get_marginal_probability(const vector<std::uint64_t>&): "
                "Invalid qubit state specified. Each qubit state must be 0, 1, or "
                "StateVector::UNMEASURED.");
        }
    }

    Fp sum = 0;
    auto d_target_index = internal::convert_host_vector_to_device_view(target_index);
    auto d_target_value = internal::convert_host_vector_to_device_view(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        _dim >> target_index.size(),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
            std::uint64_t basis = i;
            for (std::uint64_t cursor = 0; cursor < d_target_index.size(); cursor++) {
                std::uint64_t insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis |= d_target_value[cursor] << insert_index;
            }
            lsum += internal::squared_norm(this->_raw[basis]);
        },
        internal::Sum<Fp, Kokkos::DefaultExecutionSpace>(sum));

    return sum;
}
FLOAT(Fp)
Fp StateVector<Fp>::get_entropy() const {
    Fp ent = 0;
    const Fp eps = static_cast<Fp>(1e-15);
    Kokkos::parallel_reduce(
        "get_entropy",
        _dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t idx, Fp & lsum) {
            Fp prob = internal::squared_norm(_raw[idx]);
            prob = (prob > eps) ? prob : eps;
            lsum += -prob * internal::log2(prob);
        },
        internal::Sum<Fp, Kokkos::DefaultExecutionSpace>(ent));
    return ent;
}
FLOAT(Fp)
void StateVector<Fp>::add_state_vector_with_coef(ComplexType coef, const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] += coef * state._raw[i]; });
    Kokkos::fence();
}
FLOAT(Fp)
void StateVector<Fp>::multiply_coef(ComplexType coef) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] *= coef; });
    Kokkos::fence();
}
FLOAT(Fp)
std::vector<std::uint64_t> StateVector<Fp>::sampling(std::uint64_t sampling_count,
                                                     std::uint64_t seed) const {
    Kokkos::View<Fp*> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        _dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & update, const bool final) {
            update += internal::squared_norm(this->_raw[i]);
            if (final) {
                stacked_prob[i + 1] = update;
            }
        });

    Kokkos::View<std::uint64_t*> result(Kokkos::ViewAllocateWithoutInitializing("result"),
                                        sampling_count);
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::parallel_for(
        sampling_count, KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            Fp r = static_cast<Fp>(rand_gen.drand(0., 1.));
            std::uint64_t lo = 0, hi = stacked_prob.size();
            while (hi - lo > 1) {
                std::uint64_t mid = (lo + hi) / 2;
                if (stacked_prob[mid] > r) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            result[i] = lo;
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    return internal::convert_device_view_to_host_vector<std::uint64_t>(result);
}
FLOAT(Fp)
void StateVector<Fp>::load(const std::vector<ComplexType>& other) {
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: StateVector::load(const vector<ComplexType>&): invalid "
            "length of state");
    }
    _raw = internal::convert_host_vector_to_device_view(other);
}
FLOAT(Fp)
StateVector<Fp> StateVector<Fp>::copy() const {
    StateVector new_vec(_n_qubits);
    Kokkos::deep_copy(new_vec._raw, _raw);
    return new_vec;
}
FLOAT(Fp)
std::string StateVector<Fp>::to_string() const {
    std::stringstream os;
    auto amp = this->get_amplitudes();
    os << "Qubit Count : " << _n_qubits << '\n';
    os << "Dimension : " << _dim << '\n';
    os << "State vector : \n";
    for (std::uint64_t i = 0; i < _dim; ++i) {
        os << "  " <<
            [](std::uint64_t n, std::uint64_t len) {
                std::string tmp;
                while (len--) {
                    tmp += ((n >> len) & 1) + '0';
                }
                return tmp;
            }(i, _n_qubits)
           << " : " << amp[i] << std::endl;
    }
    return os.str();
}

FLOAT_DECLARE_CLASS(StateVector)

}  // namespace scaluq
