#include <scaluq/state/state_vector.hpp>
#include <scaluq/util/utility.hpp>
#include <type_traits>

#include "../prec_space.hpp"
#include "../util/math.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space>::StateVector(std::uint64_t n_qubits)
    : _n_qubits(n_qubits),
      _dim(1ULL << n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim) {
    set_zero_state();
}
template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space>::StateVector(Kokkos::View<ComplexType*, internal::SpaceType<Space>> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::set_amplitude_at(std::uint64_t index, StdComplex c) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
}
template <Precision Prec, ExecutionSpace Space>
StdComplex StateVector<Prec, Space>::get_amplitude_at(std::uint64_t index) {
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
    ComplexType val = host_view();
    return StdComplex(static_cast<double>(val.real()), static_cast<double>(val.imag()));
}
template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space> StateVector<Prec, Space>::Haar_random_state(std::uint64_t n_qubits,
                                                                     std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<internal::SpaceType<Space>> rand_pool(seed);
    auto state(StateVector<Prec, Space>::uninitialized_state(n_qubits));
    Kokkos::parallel_for(
        "Haar_random_state",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, state._dim),
        KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            state._raw(i) = ComplexType(static_cast<FloatType>(rand_gen.normal(0.0, 1.0)),
                                        static_cast<FloatType>(rand_gen.normal(0.0, 1.0)));
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    state.normalize();
    return state;
}
template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space> StateVector<Prec, Space>::uninitialized_state(std::uint64_t n_qubits) {
    StateVector<Prec, Space> state;
    state._n_qubits = n_qubits;
    state._dim = 1ULL << n_qubits;
    state._raw = Kokkos::View<ComplexType*, internal::SpaceType<Space>>(
        Kokkos::ViewAllocateWithoutInitializing("state"), state._dim);
    return state;
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::set_zero_state() {
    Kokkos::parallel_for(
        "set_zero_state",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _raw[i] = (i == 0); });
    Kokkos::fence();
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::set_zero_norm_state() {
    Kokkos::deep_copy(_raw, 0);
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::set_computational_basis(std::uint64_t basis) {
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(std::uint64_t): "
            "index of computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::parallel_for(
        "set_computational_basis",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _raw[i] = (i == basis); });
    Kokkos::fence();
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    *this = Haar_random_state(n_qubits, seed);
}
template <Precision Prec, ExecutionSpace Space>
[[nodiscard]] std::vector<StdComplex> StateVector<Prec, Space>::get_amplitudes() const {
    std::vector<ComplexType> v = internal::convert_view_to_vector<ComplexType, Space>(_raw);
    std::vector<StdComplex> ret(_dim);
    std::ranges::transform(v, ret.begin(), [&](const ComplexType& c) {
        return StdComplex(static_cast<double>(c.real()), static_cast<double>(c.imag()));
    });
    return ret;
}
template <Precision Prec, ExecutionSpace Space>
double StateVector<Prec, Space>::get_squared_norm() const {
    FloatType norm;
    Kokkos::parallel_reduce(
        "get_squared_norm",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t it, FloatType & tmp) {
            tmp += internal::squared_norm<Prec>(this->_raw[it]);
        },
        norm);
    return static_cast<double>(norm);
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::normalize() {
    const FloatType norm = internal::sqrt(static_cast<FloatType>(this->get_squared_norm()));
    Kokkos::parallel_for(
        "normalize",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t it) { this->_raw[it] /= norm; });
    Kokkos::fence();
}
template <Precision Prec, ExecutionSpace Space>
double StateVector<Prec, Space>::get_zero_probability(std::uint64_t target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(std::uint64_t): index "
            "of target qubit must be smaller than qubit_count");
    }
    FloatType sum = 0;
    Kokkos::parallel_reduce(
        "zero_prob",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim >> 1),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += internal::squared_norm(this->_raw[basis_0]);
        },
        sum);
    return static_cast<double>(sum);
}
template <Precision Prec, ExecutionSpace Space>
double StateVector<Prec, Space>::get_marginal_probability(
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

    FloatType sum = 0;
    auto d_target_index = internal::convert_vector_to_view<std::uint64_t, Space>(target_index);
    auto d_target_value = internal::convert_vector_to_view<std::uint64_t, Space>(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim >> target_index.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
            std::uint64_t basis = i;
            for (std::uint64_t cursor = 0; cursor < d_target_index.size(); cursor++) {
                std::uint64_t insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis |= d_target_value[cursor] << insert_index;
            }
            lsum += internal::squared_norm(this->_raw[basis]);
        },
        sum);

    return static_cast<double>(sum);
}
template <Precision Prec, ExecutionSpace Space>
double StateVector<Prec, Space>::get_entropy() const {
    FloatType ent = 0;
    Kokkos::parallel_reduce(
        "get_entropy",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t idx, FloatType & lsum) {
            FloatType prob = internal::squared_norm(_raw[idx]);
            if (prob > FloatType{0}) {
                lsum += -prob * internal::log2(prob);
            }
        },
        ent);
    return ent;
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::add_state_vector_with_coef(StdComplex coef,
                                                          const StateVector& state) {
    Kokkos::parallel_for(
        "add_state_vector_with_coef",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) {
            this->_raw[i] += ComplexType(coef) * state._raw[i];
        });
    Kokkos::fence();
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::multiply_coef(StdComplex coef) {
    Kokkos::parallel_for(
        "multiply_coef",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] *= coef; });
    Kokkos::fence();
}
template <Precision Prec, ExecutionSpace Space>
std::vector<std::uint64_t> StateVector<Prec, Space>::sampling(std::uint64_t sampling_count,
                                                              std::uint64_t seed) const {
    Kokkos::View<FloatType*, internal::SpaceType<Space>> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "sampling (compute stacked prob)",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, _dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & update, const bool final) {
            update += internal::squared_norm(this->_raw[i]);
            if (final) {
                stacked_prob[i + 1] = update;
            }
        });

    Kokkos::Random_XorShift64_Pool<internal::SpaceType<Space>> rand_pool(seed);
    std::vector<std::uint64_t> result(sampling_count);
    std::vector<std::uint64_t> todo(sampling_count);
    std::iota(todo.begin(), todo.end(), 0);
    while (!todo.empty()) {
        std::size_t todo_count = todo.size();
        Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> result_buf(
            Kokkos::ViewAllocateWithoutInitializing("result_buf"), todo_count);
        Kokkos::parallel_for(
            "sampling (choose)",
            Kokkos::RangePolicy<internal::SpaceType<Space>>(0, todo_count),
            KOKKOS_LAMBDA(std::uint64_t i) {
                auto rand_gen = rand_pool.get_state();
                FloatType r = static_cast<FloatType>(rand_gen.drand(0., 1.));
                std::uint64_t lo = 0, hi = stacked_prob.size();
                while (hi - lo > 1) {
                    std::uint64_t mid = (lo + hi) / 2;
                    if (stacked_prob[mid] > r) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                result_buf[i] = lo;
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();
        auto result_buf_host = internal::convert_view_to_vector<std::uint64_t, Space>(result_buf);
        // Especially for F16 and BF16, sampling sometimes fails with result == _dim.
        // In this case, re-sampling is performed.
        std::vector<std::uint64_t> next_todo;
        for (std::size_t i = 0; i < todo_count; i++) {
            if (result_buf_host[i] == _dim) {
                next_todo.push_back(todo[i]);
            } else {
                result[todo[i]] = result_buf_host[i];
            }
        }
        todo.swap(next_todo);
    }
    return result;
}
template <Precision Prec, ExecutionSpace Space>
void StateVector<Prec, Space>::load(const std::vector<StdComplex>& other) {
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: StateVector::load(const vector<ComplexType>&): invalid "
            "length of state");
    }
    std::vector<ComplexType> other_complex(_dim);
    std::ranges::copy(other, other_complex.begin());
    _raw = internal::convert_vector_to_view<ComplexType, Space>(other_complex);
}
template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space> StateVector<Prec, Space>::copy() const {
    StateVector<Prec, Space> new_vec(_n_qubits);
    Kokkos::deep_copy(new_vec._raw, _raw);
    return new_vec;
}
template <Precision Prec, ExecutionSpace Space>
std::string StateVector<Prec, Space>::to_string() const {
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

template class StateVector<internal::Prec, internal::Space>;

}  // namespace scaluq
