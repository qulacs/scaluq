#include <scaluq/state/state_vector.hpp>
#include <type_traits>

#include "../util/math.hpp"
#include "../util/template.hpp"

namespace scaluq {
<<<<<<< HEAD
template <Precision Prec>
StateVector<Prec>::StateVector(std::uint64_t n_qubits)
=======
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp>::StateVector(std::uint64_t n_qubits)
>>>>>>> set-space
    : _n_qubits(n_qubits),
      _dim(1ULL << n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim) {
    set_zero_state();
}
<<<<<<< HEAD
template <Precision Prec>
StateVector<Prec>::StateVector(Kokkos::View<ComplexType*> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}
template <Precision Prec>
void StateVector<Prec>::set_amplitude_at(std::uint64_t index, StdComplex c) {
=======
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp>::StateVector(Kokkos::View<ComplexType*, Sp> view)
    : _n_qubits(std::bit_width(view.extent(0)) - 1), _dim(view.extent(0)), _raw(view) {}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_amplitude_at(std::uint64_t index, ComplexType c) {
>>>>>>> set-space
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
}
<<<<<<< HEAD
template <Precision Prec>
StdComplex StateVector<Prec>::get_amplitude_at(std::uint64_t index) {
=======
FLOAT_AND_SPACE(Fp, Sp)
Complex<Fp> StateVector<Fp, Sp>::get_amplitude_at(std::uint64_t index) {
>>>>>>> set-space
    Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
    ComplexType val = host_view();
    return StdComplex(static_cast<double>(val.real()), static_cast<double>(val.imag()));
}
<<<<<<< HEAD
template <Precision Prec>
StateVector<Prec> StateVector<Prec>::Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVector<Prec> state(n_qubits);
=======
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp> StateVector<Fp, Sp>::Haar_random_state(std::uint64_t n_qubits,
                                                           std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<Sp> rand_pool(seed);
    StateVector<Fp, Sp> state(n_qubits);
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, state._dim), KOKKOS_LAMBDA(std::uint64_t i) {
            auto rand_gen = rand_pool.get_state();
            state._raw(i) = ComplexType(static_cast<FloatType>(rand_gen.normal(0.0, 1.0)),
                                        static_cast<FloatType>(rand_gen.normal(0.0, 1.0)));
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    state.normalize();
    return state;
}
<<<<<<< HEAD
template <Precision Prec>
void StateVector<Prec>::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at(0, 1);
}
template <Precision Prec>
void StateVector<Prec>::set_zero_norm_state() {
    Kokkos::deep_copy(_raw, 0);
}
template <Precision Prec>
void StateVector<Prec>::set_computational_basis(std::uint64_t basis) {
=======
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at(0, 1);
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::set_computational_basis(std::uint64_t basis) {
>>>>>>> set-space
    if (basis >= _dim) {
        throw std::runtime_error(
            "Error: StateVector::set_computational_basis(std::uint64_t): "
            "index of computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at(basis, 1);
}
<<<<<<< HEAD
template <Precision Prec>
void StateVector<Prec>::set_Haar_random_state(std::uint64_t n_qubits, std::uint64_t seed) {
    *this = Haar_random_state(n_qubits, seed);
}
template <Precision Prec>
[[nodiscard]] std::vector<StdComplex> StateVector<Prec>::get_amplitudes() const {
    std::vector<ComplexType> v = internal::convert_device_view_to_host_vector<ComplexType>(_raw);
    std::vector<StdComplex> ret(_dim);
    std::ranges::transform(v, ret.begin(), [&](const ComplexType& c) {
        return StdComplex(static_cast<double>(c.real()), static_cast<double>(c.imag()));
    });
    return ret;
}
template <Precision Prec>
double StateVector<Prec>::get_squared_norm() const {
    FloatType norm;
    Kokkos::parallel_reduce(
        this->_dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t it, FloatType & tmp) {
=======
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
>>>>>>> set-space
            tmp += internal::squared_norm(this->_raw[it]);
        },
        internal::Sum<FloatType, Kokkos::DefaultExecutionSpace>(norm));
    return static_cast<double>(norm);
}
<<<<<<< HEAD
template <Precision Prec>
void StateVector<Prec>::normalize() {
    const FloatType norm = internal::sqrt(static_cast<FloatType>(this->get_squared_norm()));
=======
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::normalize() {
    const Fp norm = std::sqrt(this->get_squared_norm());
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t it) { this->_raw[it] /= norm; });
    Kokkos::fence();
}
<<<<<<< HEAD
template <Precision Prec>
double StateVector<Prec>::get_zero_probability(std::uint64_t target_qubit_index) const {
=======
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_zero_probability(std::uint64_t target_qubit_index) const {
>>>>>>> set-space
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVector::get_zero_probability(std::uint64_t): index "
            "of target qubit must be smaller than qubit_count");
    }
    FloatType sum = 0;
    Kokkos::parallel_reduce(
        "zero_prob",
<<<<<<< HEAD
        _dim >> 1,
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
=======
        Kokkos::RangePolicy<Sp>(0, this->_dim >> 1),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
>>>>>>> set-space
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
            lsum += internal::squared_norm(this->_raw[basis_0]);
        },
        internal::Sum<FloatType, Kokkos::DefaultExecutionSpace>(sum));
    return static_cast<double>(sum);
}
<<<<<<< HEAD
template <Precision Prec>
double StateVector<Prec>::get_marginal_probability(
=======
FLOAT_AND_SPACE(Fp, Sp)
Fp StateVector<Fp, Sp>::get_marginal_probability(
>>>>>>> set-space
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

<<<<<<< HEAD
    FloatType sum = 0;
    auto d_target_index = internal::convert_host_vector_to_device_view(target_index);
    auto d_target_value = internal::convert_host_vector_to_device_view(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        _dim >> target_index.size(),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & lsum) {
=======
    Fp sum = 0.;
    auto d_target_index = internal::convert_vector_to_view<std::uint64_t, Sp>(target_index);
    auto d_target_value = internal::convert_vector_to_view<std::uint64_t, Sp>(target_value);

    Kokkos::parallel_reduce(
        "marginal_prob",
        Kokkos::RangePolicy<Sp>(0, this->_dim >> target_index.size()),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & lsum) {
>>>>>>> set-space
            std::uint64_t basis = i;
            for (std::uint64_t cursor = 0; cursor < d_target_index.size(); cursor++) {
                std::uint64_t insert_index = d_target_index[cursor];
                basis = internal::insert_zero_to_basis_index(basis, insert_index);
                basis |= d_target_value[cursor] << insert_index;
            }
            lsum += internal::squared_norm(this->_raw[basis]);
        },
        internal::Sum<FloatType, Kokkos::DefaultExecutionSpace>(sum));

    return static_cast<double>(sum);
}
<<<<<<< HEAD
template <Precision Prec>
double StateVector<Prec>::get_entropy() const {
    FloatType ent = 0;
    Kokkos::parallel_reduce(
        "get_entropy",
        _dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t idx, FloatType & lsum) {
            FloatType prob = internal::squared_norm(_raw[idx]);
            if (prob > FloatType{0}) {
                lsum += -prob * internal::log2(prob);
            }
=======
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
>>>>>>> set-space
        },
        internal::Sum<FloatType, Kokkos::DefaultExecutionSpace>(ent));
    return ent;
}
<<<<<<< HEAD
template <Precision Prec>
void StateVector<Prec>::add_state_vector_with_coef(StdComplex coef, const StateVector& state) {
    Kokkos::parallel_for(
        this->_dim, KOKKOS_CLASS_LAMBDA(std::uint64_t i) {
            this->_raw[i] += ComplexType(coef) * state._raw[i];
        });
    Kokkos::fence();
}
template <Precision Prec>
void StateVector<Prec>::multiply_coef(StdComplex coef) {
=======
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::add_state_vector_with_coef(ComplexType coef, const StateVector& state) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] += coef * state._raw[i]; });
    Kokkos::fence();
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::multiply_coef(ComplexType coef) {
>>>>>>> set-space
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { this->_raw[i] *= coef; });
    Kokkos::fence();
}
<<<<<<< HEAD
template <Precision Prec>
std::vector<std::uint64_t> StateVector<Prec>::sampling(std::uint64_t sampling_count,
                                                       std::uint64_t seed) const {
    Kokkos::View<FloatType*> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        _dim,
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, FloatType & update, const bool final) {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::vector<std::uint64_t> StateVector<Fp, Sp>::sampling(std::uint64_t sampling_count,
                                                         std::uint64_t seed) const {
    Kokkos::View<Fp*, Sp> stacked_prob("prob", _dim + 1);
    Kokkos::parallel_scan(
        "compute_stacked_prob",
        Kokkos::RangePolicy<Sp>(0, this->_dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i, Fp & update, const bool final) {
>>>>>>> set-space
            update += internal::squared_norm(this->_raw[i]);
            if (final) {
                stacked_prob[i + 1] = update;
            }
        });

<<<<<<< HEAD
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    std::vector<std::uint64_t> result(sampling_count);
    std::vector<std::uint64_t> todo(sampling_count);
    std::iota(todo.begin(), todo.end(), 0);
    while (!todo.empty()) {
        std::size_t todo_count = todo.size();
        Kokkos::View<std::uint64_t*> result_buf(
            Kokkos::ViewAllocateWithoutInitializing("result_buf"), todo_count);
        Kokkos::parallel_for(
            todo_count, KOKKOS_LAMBDA(std::uint64_t i) {
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
=======
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
>>>>>>> set-space
                }
                result_buf[i] = lo;
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();
        auto result_buf_host =
            internal::convert_device_view_to_host_vector<std::uint64_t>(result_buf);
        // Especially for F16 and BF16, sampling sometimes fails with result == _dim.
        // In this case, re-sampling is performed.
        std::vector<std::uint64_t> next_todo;
        for (std::size_t i = 0; i < todo_count; i++) {
            if (result_buf_host[i] == _dim) {
                next_todo.push_back(todo[i]);
            } else {
                result[todo[i]] = result_buf_host[i];
            }
<<<<<<< HEAD
        }
        todo.swap(next_todo);
    }
    return result;
}
template <Precision Prec>
void StateVector<Prec>::load(const std::vector<StdComplex>& other) {
=======
            result[i] = lo;
            rand_pool.free_state(rand_gen);
        });
    Kokkos::fence();
    return internal::convert_view_to_vector<std::uint64_t, Sp>(result);
}
FLOAT_AND_SPACE(Fp, Sp)
void StateVector<Fp, Sp>::load(const std::vector<Complex<Fp>>& other) {
>>>>>>> set-space
    if (other.size() != _dim) {
        throw std::runtime_error(
            "Error: StateVector::load(const vector<ComplexType>&): invalid "
            "length of state");
    }
<<<<<<< HEAD
    std::vector<ComplexType> other_complex(_dim);
    std::ranges::copy(other, other_complex.begin());
    _raw = internal::convert_host_vector_to_device_view(other_complex);
}
template <Precision Prec>
StateVector<Prec> StateVector<Prec>::copy() const {
=======
    _raw = internal::convert_vector_to_view<Complex<Fp>, Sp>(other);
}
FLOAT_AND_SPACE(Fp, Sp)
StateVector<Fp, Sp> StateVector<Fp, Sp>::copy() const {
>>>>>>> set-space
    StateVector new_vec(_n_qubits);
    Kokkos::deep_copy(new_vec._raw, _raw);
    return new_vec;
}
<<<<<<< HEAD
template <Precision Prec>
std::string StateVector<Prec>::to_string() const {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::string StateVector<Fp, Sp>::to_string() const {
>>>>>>> set-space
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

<<<<<<< HEAD
SCALUQ_DECLARE_CLASS_FOR_PRECISION(StateVector)
=======
FLOAT_AND_SPACE_DECLARE_CLASS(StateVector)
>>>>>>> set-space

}  // namespace scaluq
