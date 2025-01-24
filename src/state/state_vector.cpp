#include <scaluq/state/state_vector.hpp>
#include <type_traits>

#include "../util/template.hpp"

namespace scaluq {
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp>::StateVector(std::uint64_t n_qubits)
    : _n_qubits(n_qubits),
      _dim(1ULL << n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim) {
    set_zero_state();
}
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp>::StateVector(Kokkos::View<ComplexType*, Sp> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_amplitude_at(std::uint64_t index, ComplexType c) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
}
FLOAT_AND_SPACE(Fp, Sp)
Complex<Fp> StateVector<Fp, Sp>::get_amplitude_at(std::uint64_t index) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
    return host_view();
}
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp> StateVector<Fp, Sp>::Haar_random_state(std::uint64_t n_qubits,
                                                           std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<Sp> rand_pool(seed);
    StateVector<Fp, Sp> state(n_qubits);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state._dim), KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            state._raw(i) = ComplexType(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    state.normalize();
    return state;
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at(0, 1);
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_computational_basis(std::uint64_t basis) {
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(std::uint64_t): "
            "index of computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at(basis, 1);
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    *this = Haar_random_state(n_qubits, seed);
}
FLOAT_AND_SPACE(Fp, Sp)
[[nodiscard]] std::vector<Complex<Fp>> StateVector<Fp, Sp>::get_amplitudes() const {
    if constexpr (std::is_same_v<Sp, DefaultSpace>) {
        return internal::convert_view_to_vector<ComplexType>(_raw);
    } else {
        return std::vector<Complex<Fp>>(_raw.data(), _raw.data() + _raw.size());
    }
}
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_squared_norm() const {
    Fp norm;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t it, Fp & tmp) {
            tmp += internal::squared_norm(this->_raw[it]);
        },
        norm);
    return norm;
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::normalize() {
    const Fp norm = std::sqrt(this->get_squared_norm());
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t it) { this->_raw[it] /= norm; });
    Kokkos::fence();
}
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_zero_probability(std::uint64_t target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(std::uint64_t): index "
            "of target qubit must be smaller than qubit_count");
    }
    Fp sum = 0.;
    Kokkos::parallel_reduce(
        "zero_prob",
        Kokkos::RangePolicy<Sp>(0, this->_dim >> 1),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += internal::squared_norm(this->_raw[basis_0]);
        },
        sum);
    return sum;
}
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_marginal_probability(
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

    Fp sum = 0.;
    auto d_target_index = internal::convert_vector_to_view<std::uint64_t, Sp>(target_index);
    auto d_target_value = internal::convert_vector_to_view<std::uint64_t, Sp>(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<Sp>(0, this->_dim >> target_index.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
            std::uint64_t basis = i;
            for (std::uint64_t cursor = 0; cursor < d_target_index.size(); cursor++) {
                std::uint64_t insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis |= d_target_value[cursor] << insert_index;
            }
            lsum += internal::squared_norm(this->_raw[basis]);
        },
        sum);

    return sum;
}
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_entropy() const {
    Fp ent = 0;
    const Fp eps = 1e-15;
    Kokkos::parallel_reduce(
        "get_entropy",
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t idx, Fp & lsum) {
            Fp prob = internal::squared_norm(_raw[idx]);
            prob = (prob > eps) ? prob : eps;
            lsum += -prob * Kokkos::log2(prob);
        },
        ent);
    return ent;
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::add_state_vector_with_coef(ComplexType coef, const StateVector& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] += coef * state._raw[i]; });
    Kokkos::fence();
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::multiply_coef(ComplexType coef) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] *= coef; });
    Kokkos::fence();
}
FLOAT_AND_SPACE(Fp, Sp)
std::vector<std::uint64_t> StateVector<Fp, Sp>::sampling(std::uint64_t sampling_count,
                                                         std::uint64_t seed) const {
    Kokkos::View<Fp*, Sp> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & update, const bool final) {
            update += internal::squared_norm(this->_raw[i]);
            if (final) {
                stacked_prob[i + 1] = update;
            }
        });

    Kokkos::View<std::uint64_t*, Sp> result(Kokkos::ViewAllocateWithoutInitializing("result"),
                                            sampling_count);
    Kokkos::Random_XorShift64_Pool<Sp> rand_pool(seed);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, sampling_count), KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            Fp r = rand_gen.drand(0., 1.);
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
    return internal::convert_view_to_vector<std::uint64_t, Sp>(result);
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::load(const std::vector<Complex<Fp>>& other) {
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: StateVector::load(const vector<ComplexType>&): invalid "
            "length of state");
    }
    _raw = internal::convert_vector_to_view<Complex<Fp>, Sp>(other);
}
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp> StateVector<Fp, Sp>::copy() const {
    StateVector new_vec(_n_qubits);
    Kokkos::deep_copy(new_vec._raw, _raw);
    return new_vec;
}
FLOAT_AND_SPACE(Fp, Sp)
std::string StateVector<Fp, Sp>::to_string() const {
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

FLOAT_AND_SPACE_DECLARE_CLASS(StateVector)

}  // namespace scaluq
