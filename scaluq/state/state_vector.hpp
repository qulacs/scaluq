#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "../util/utility.hpp"

namespace scaluq {

struct Host {};
struct Default {};

template <typename T>
struct ExecutionSpaceMapper {
    static_assert(std::is_same<T, Host>::value || std::is_same<T, Default>::value,
                  "Unsupported execution space tag");
};
template <>
struct ExecutionSpaceMapper<Host> {
    using Type = Kokkos::DefaultHostExecutionSpace;
};
template <>
struct ExecutionSpaceMapper<Default> {
    using Type = Kokkos::DefaultExecutionSpace;
};

template <std::floating_point FloatType = double, typename Space = Default>
class StateVector {
    UINT _n_qubits;
    UINT _dim;
    using ExecutionSpace = ExecutionSpaceMapper<Space>::Type;
    using ComplexType = Complex<FloatType>;

public:
    static constexpr UINT UNMEASURED = 2;
    Kokkos::View<ComplexType*, ExecutionSpace> _raw;
    StateVector() = default;
    StateVector(UINT n_qubits)
        : _n_qubits(n_qubits),
          _dim(1ULL << n_qubits),
          _raw(Kokkos::ViewAllocateWithoutInitializing("state"), this->_dim) {
        set_zero_state();
    }
    StateVector(const StateVector& other) = default;

    StateVector& operator=(const StateVector& other) = default;

    /**
     * @attention Very slow. You should use load() instead if you can.
     */
    void set_amplitude_at_index(UINT index, ComplexType c) {
        Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
        host_view() = c;
        Kokkos::deep_copy(Kokkos::subview(_raw, index), host_view());
    }

    /**
     * @attention Very slow. You should use amplitudes() instead if you can.
     */
    [[nodiscard]] ComplexType get_amplitude_at_index(UINT index) const {
        Kokkos::View<ComplexType, Kokkos::HostSpace> host_view("single_value");
        Kokkos::deep_copy(host_view, Kokkos::subview(_raw, index));
        return host_view();
    }

    [[nodiscard]] static StateVector Haar_random_state(UINT n_qubits,
                                                       UINT seed = std::random_device()()) {
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
        StateVector<FloatType, Space> state(n_qubits);
        Kokkos::parallel_for(
            state._dim, KOKKOS_LAMBDA(UINT i) {
                auto rand_gen = rand_pool.get_state();
                state._raw(i) = ComplexType(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();
        state.normalize();
        return state;
    }

    /**
     * @brief zero-fill
     */
    void set_zero_state() {
        Kokkos::deep_copy(_raw, 0);
        set_amplitude_at_index(0, 1);
    }
    void set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }
    void set_computational_basis(UINT basis) {
        if (basis >= _dim) {
            throw std::runtime_error(
                "Error: StateVector::set_computational_basis(UINT): "
                "index of "
                "computational basis must be smaller than 2^qubit_count");
        }
        Kokkos::deep_copy(_raw, 0);
        set_amplitude_at_index(basis, 1);
    }

    [[nodiscard]] UINT n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] UINT dim() const { return this->_dim; }

    [[nodiscard]] std::vector<ComplexType> amplitudes() const {
        return internal::convert_device_view_to_host_vector(_raw);
    }

    [[nodiscard]] FloatType get_squared_norm() const {
        FloatType norm;
        Kokkos::parallel_reduce(
            this->_dim,
            KOKKOS_CLASS_LAMBDA(UINT it, FloatType & tmp) {
                tmp += internal::squared_norm(this->_raw[it]);
            },
            norm);
        return norm;
    }

    void normalize() {
        const FloatType norm = std::sqrt(this->get_squared_norm());
        Kokkos::parallel_for(
            this->_dim, KOKKOS_CLASS_LAMBDA(UINT it) { this->_raw[it] /= norm; });
        Kokkos::fence();
    }

    [[nodiscard]] FloatType get_zero_probability(UINT target_qubit_index) const {
        if (target_qubit_index >= _n_qubits) {
            throw std::runtime_error(
                "Error: StateVector::get_zero_probability(UINT): index "
                "of target qubit must be smaller than qubit_count");
        }
        FloatType sum = 0.;
        Kokkos::parallel_reduce(
            "zero_prob",
            _dim >> 1,
            KOKKOS_CLASS_LAMBDA(UINT i, FloatType & lsum) {
                UINT basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
                lsum += internal::squared_norm(this->_raw[basis_0]);
            },
            sum);
        return sum;
    }

    [[nodiscard]] FloatType get_marginal_probability(
        const std::vector<UINT>& measured_values) const {
        if (measured_values.size() != _n_qubits) {
            throw std::runtime_error(
                "Error: "
                "StateVector::get_marginal_probability(const vector<UINT>&): "
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
                    "Error: "
                    "StateVector::get_marginal_probability(const vector<UINT>&): Invalid qubit "
                    "state "
                    "specified. Each qubit state must be 0, 1, or "
                    "StateVector::UNMEASURED.");
            }
        }

        FloatType sum = 0.;
        auto d_target_index = internal::convert_host_vector_to_device_view(target_index);
        auto d_target_value = internal::convert_host_vector_to_device_view(target_value);

        Kokkos::parallel_reduce(
            "marginal_prob",
            _dim >> target_index.size(),
            KOKKOS_CLASS_LAMBDA(UINT i, FloatType & lsum) {
                UINT basis = i;
                for (UINT cursor = 0; cursor < d_target_index.size(); cursor++) {
                    UINT insert_index = d_target_index[cursor];
                    basis = internal::insert_zero_to_basis_index(basis, insert_index);
                    basis |= d_target_value[cursor] << insert_index;
                }
                lsum += internal::squared_norm(this->_raw[basis]);
            },
            sum);

        return sum;
    }
    [[nodiscard]] FloatType get_entropy() const {
        FloatType ent = 0;
        const FloatType eps = 1e-15;
        Kokkos::parallel_reduce(
            "get_entropy",
            _dim,
            KOKKOS_CLASS_LAMBDA(UINT idx, FloatType & lsum) {
                FloatType prob = internal::squared_norm(_raw[idx]);
                prob = (prob > eps) ? prob : eps;
                lsum += -prob * Kokkos::log(prob);
            },
            ent);
        return ent;
    }

    void add_state_vector(const StateVector& state) {
        Kokkos::parallel_for(
            this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] += state._raw[i]; });
        Kokkos::fence();
    }
    void add_state_vector_with_coef(ComplexType coef, const StateVector& state) {
        Kokkos::parallel_for(
            this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] += coef * state._raw[i]; });
        Kokkos::fence();
    }
    void multiply_coef(ComplexType coef) {
        Kokkos::parallel_for(
            this->_dim, KOKKOS_CLASS_LAMBDA(UINT i) { this->_raw[i] *= coef; });
        Kokkos::fence();
    }

    [[nodiscard]] std::vector<UINT> sampling(UINT sampling_count,
                                             UINT seed = std::random_device()()) const {
        Kokkos::View<FloatType*> stacked_prob("prob", _dim + 1);
        Kokkos::parallel_scan(
            "compute_stacked_prob",
            _dim,
            KOKKOS_CLASS_LAMBDA(UINT i, FloatType & update, const bool final) {
                update += internal::squared_norm(this->_raw[i]);
                if (final) {
                    stacked_prob[i + 1] = update;
                }
            });

        Kokkos::View<UINT*> result(Kokkos::ViewAllocateWithoutInitializing("result"),
                                   sampling_count);
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
        Kokkos::parallel_for(
            sampling_count, KOKKOS_LAMBDA(UINT i) {
                auto rand_gen = rand_pool.get_state();
                FloatType r = rand_gen.drand(0., 1.);
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
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector(result);
    }

    void load(const std::vector<ComplexType>& other) {
        if (other.size() != _dim) {
            throw std::runtime_error(
                "Error: StateVector::load(const vector<ComplexType>&): invalid "
                "length of state");
        }
        _raw = internal::convert_host_vector_to_device_view(other);
    }

    [[nodiscard]] StateVector copy() const {
        StateVector new_vec(_n_qubits);
        Kokkos::deep_copy(new_vec._raw, _raw);
        return new_vec;
    }

    friend std::ostream& operator<<(std::ostream& os, const StateVector& state) {
        os << state.to_string();
        return os;
    }

    [[nodiscard]] std::string to_string() const {
        std::stringstream os;
        auto amp = this->amplitudes();
        os << " *** Quantum State ***\n";
        os << " * Qubit Count : " << _n_qubits << '\n';
        os << " * Dimension   : " << _dim << '\n';
        os << " * State vector : \n";
        for (UINT i = 0; i < _dim; ++i) {
            os <<
                [](UINT n, UINT len) {
                    std::string tmp;
                    while (len--) {
                        tmp += ((n >> len) & 1) + '0';
                    }
                    return tmp;
                }(i, _n_qubits)
               << ": " << amp[i] << std::endl;
        }
        return os.str();
    }
};
}  // namespace scaluq
