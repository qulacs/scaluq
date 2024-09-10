#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <std::floating_point FloatType = double, typename Space = DefaultSpace>
class StateVectorBatched {
    std::uint64_t _batch_size;
    std::uint64_t _n_qubits;
    std::uint64_t _dim;
    using ComplexType = Kokkos::complex<FloatType>;

    static_assert(std::is_same_v<Space, HostSpace> || std::is_same_v<Space, DefaultSpace>,
                  "Unsupported execution space tag");

public:
    Kokkos::View<ComplexType**, Kokkos::LayoutRight> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(std::uint64_t batch_size, std::uint64_t n_qubits)
        : _batch_size(batch_size),
          _n_qubits(n_qubits),
          _dim(1ULL << _n_qubits),
          _raw(Kokkos::View<ComplexType**, Kokkos::LayoutRight>(
              Kokkos::ViewAllocateWithoutInitializing("states"), _batch_size, _dim)) {
        set_zero_state();
    }
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] std::uint64_t n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] std::uint64_t dim() const { return this->_dim; }

    [[nodiscard]] std::uint64_t batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector<FloatType, Space>& state) {
        if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::set_state_vector(const StateVector&): Dimensions of "
                "source and destination views do not match.");
        }
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
                _raw(batch_id, i) = state._raw(i);
            });
        Kokkos::fence();
    }

    void set_state_vector_at(std::uint64_t batch_id, const StateVector<FloatType, Space>& state) {
        if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::set_state_vector(std::uint64_t, const StateVector&): "
                "Dimensions of source and destination views do not match.");
        }
        Kokkos::parallel_for(
            _dim, KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _raw(batch_id, i) = state._raw(i); });
        Kokkos::fence();
    }

    [[nodiscard]] StateVector<FloatType, Space> get_state_vector_at(std::uint64_t batch_id) const {
        StateVector<FloatType, Space> ret(_n_qubits);
        Kokkos::parallel_for(
            _dim, KOKKOS_CLASS_LAMBDA(std::uint64_t i) { ret._raw(i) = _raw(batch_id, i); });
        Kokkos::fence();
        return ret;
    }

    void set_zero_state() { set_computational_basis(0); }

    void set_computational_basis(std::uint64_t basis) {
        if (basis >= _dim) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::set_computational_basis(std::uint64_t): "
                "index of computational basis must be smaller than 2^qubit_count");
        }
        Kokkos::deep_copy(_raw, 0);
        Kokkos::parallel_for(
            _batch_size, KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _raw(i, basis) = 1; });
        Kokkos::fence();
    }

    void set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }

    [[nodiscard]] std::vector<std::vector<std::uint64_t>> sampling(
        std::uint64_t sampling_count, std::uint64_t seed = std::random_device()()) const {
        Kokkos::View<FloatType**> stacked_prob("prob", _batch_size, _dim + 1);

        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, _dim),
                                      [&](std::uint64_t i, FloatType& update, const bool final) {
                                          update += internal::squared_norm(this->_raw(batch_id, i));
                                          if (final) {
                                              stacked_prob(batch_id, i + 1) = update;
                                          }
                                      });
            });
        Kokkos::fence();

        Kokkos::View<std::uint64_t**> result(
            Kokkos::ViewAllocateWithoutInitializing("result"), _batch_size, sampling_count);
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, sampling_count}),
            KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
                auto rand_gen = rand_pool.get_state();
                FloatType r = rand_gen.drand(0., 1.);
                std::uint64_t lo = 0, hi = stacked_prob.extent(1);
                while (hi - lo > 1) {
                    std::uint64_t mid = (lo + hi) / 2;
                    if (stacked_prob(batch_id, mid) > r) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                result(batch_id, i) = lo;
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();

        auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
        std::vector<std::vector<std::uint64_t>> vv(result.extent(0),
                                                   std::vector<std::uint64_t>(result.extent(1), 0));
        for (size_t i = 0; i < result.extent(0); ++i) {
            for (size_t j = 0; j < result.extent(1); ++j) {
                vv[i][j] = view_h(i, j);
            }
        }
        return vv;
    }

    [[nodiscard]] static StateVectorBatched Haar_random_states(
        std::uint64_t batch_size,
        std::uint64_t n_qubits,
        bool set_same_state,
        std::uint64_t seed = std::random_device()()) {
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
        StateVectorBatched states(batch_size, n_qubits);
        if (set_same_state) {
            states.set_state_vector(
                StateVector<FloatType, Space>::Haar_random_state(n_qubits, seed));
        } else {
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
                KOKKOS_LAMBDA(std::uint64_t b, std::uint64_t i) {
                    auto rand_gen = rand_pool.get_state();
                    states._raw(b, i) =
                        ComplexType(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
                    rand_pool.free_state(rand_gen);
                });
            Kokkos::fence();
            states.normalize();
        }
        return states;
    }

    [[nodiscard]] std::vector<std::vector<ComplexType>> get_amplitudes() const {
        std::vector<std::vector<ComplexType>> vv(_raw.extent(0),
                                                 std::vector<ComplexType>(_raw.extent(1), 0));
        for (size_t i = 0; i < _raw.extent(0); ++i) {
            for (size_t j = 0; j < _raw.extent(1); ++j) {
                vv[i][j] = _raw(i, j);
            }
        }
        return vv;
    }

    [[nodiscard]] std::vector<FloatType> get_squared_norm() const {
        Kokkos::View<FloatType*> norms(Kokkos::ViewAllocateWithoutInitializing("norms"),
                                       _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                FloatType nrm = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [&](const std::uint64_t& i, FloatType& lcl) {
                        lcl += internal::squared_norm(_raw(batch_id, i));
                    },
                    nrm);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { norms[batch_id] = nrm; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector<FloatType>(norms);
    }

    void normalize() {
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                FloatType nrm = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [&](const std::uint64_t& i, FloatType& lcl) {
                        lcl += internal::squared_norm(_raw(batch_id, i));
                    },
                    nrm);
                team.team_barrier();
                nrm = Kokkos::sqrt(nrm);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, _dim),
                                     [&](const std::uint64_t& i) { _raw(batch_id, i) /= nrm; });
            });
        Kokkos::fence();
    }

    [[nodiscard]] std::vector<FloatType> get_zero_probability(
        std::uint64_t target_qubit_index) const {
        if (target_qubit_index >= _n_qubits) {
            throw std::runtime_error(
                "Error: StateVectorBatched::get_zero_probability(std::uint64_t): index "
                "of target qubit must be smaller than qubit_count");
        }
        Kokkos::View<FloatType*> probs("probs", _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                FloatType sum = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim >> 1),
                    [&](std::uint64_t i, FloatType& lsum) {
                        std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(i, target_qubit_index);
                        lsum += internal::squared_norm(_raw(batch_id, basis_0));
                    },
                    sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector<FloatType>(probs);
    }

    [[nodiscard]] std::vector<FloatType> get_marginal_probability(
        const std::vector<std::uint64_t>& measured_values) const {
        if (measured_values.size() != _n_qubits) {
            throw std::runtime_error(
                "Error: StateVectorBatched::get_marginal_probability(const "
                "vector<std::uint64_t>&): the length of measured_values must be equal to "
                "qubit_count");
        }

        std::vector<std::uint64_t> target_index;
        std::vector<std::uint64_t> target_value;
        for (std::uint64_t i = 0; i < measured_values.size(); ++i) {
            std::uint64_t measured_value = measured_values[i];
            if (measured_value == 0 || measured_value == 1) {
                target_index.push_back(i);
                target_value.push_back(measured_value);
            } else if (measured_value != StateVector<FloatType, Space>::UNMEASURED) {
                throw std::runtime_error(
                    "Error:StateVectorBatched::get_marginal_probability(const "
                    "vector<std::uint64_t>&): Invalid qubit state specified. Each qubit state must "
                    "be 0, 1, or StateVector::UNMEASURED.");
            }
        }

        auto target_index_d = internal::convert_host_vector_to_device_view(target_index);
        auto target_value_d = internal::convert_host_vector_to_device_view(target_value);
        Kokkos::View<FloatType*> probs("probs", _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                FloatType sum = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim >> target_index_d.size()),
                    [&](std::uint64_t i, FloatType& lsum) {
                        std::uint64_t basis = i;
                        for (std::uint64_t cursor = 0; cursor < target_index_d.size(); cursor++) {
                            std::uint64_t insert_index = target_index_d[cursor];
                            basis = internal::insert_zero_to_basis_index(basis, insert_index);
                            basis ^= target_value_d[cursor] << insert_index;
                        }
                        lsum += internal::squared_norm(_raw(batch_id, basis));
                    },
                    sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector<FloatType>(probs);
    }

    [[nodiscard]] std::vector<FloatType> get_entropy() const {
        Kokkos::View<FloatType*> ents("ents", _batch_size);
        const FloatType eps = 1e-15;
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                FloatType sum = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [&](std::uint64_t idx, FloatType& lsum) {
                        FloatType prob = internal::squared_norm(_raw(batch_id, idx));
                        prob = Kokkos::max(prob, eps);
                        lsum += -prob * Kokkos::log(prob);
                    },
                    sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { ents[batch_id] = sum; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector(ents);
    }

    void add_state_vector_with_coef(const ComplexType& coef, const StateVectorBatched& states) {
        if (n_qubits() != states.n_qubits() || batch_size() != states.batch_size()) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::add_state_vector(const StateVectorBatched&): invalid "
                "states");
        }
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
                _raw(batch_id, i) += coef * states._raw(batch_id, i);
            });
        Kokkos::fence();
    }

    void multiply_coef(const ComplexType& coef) {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
                _raw(batch_id, i) *= coef;
            });
        Kokkos::fence();
    }

    void load(const std::vector<std::vector<ComplexType>>& states) {
        if (states.size() != _batch_size) {
            throw std::runtime_error(
                "Error: StateVectorBatched::load(std::vector<std::vector<ComplexType>>&): invalid "
                "batch_size");
        }
        for (std::uint64_t b = 0; b < states.size(); ++b) {
            if (states[b].size() != _dim) {
                throw std::runtime_error(
                    "Error: StateVectorBatched::load(std::vector<std::vector<ComplexType>>&): "
                    "invalid "
                    "length of state");
            }
        }

        auto view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), _raw);
        for (std::uint64_t b = 0; b < states.size(); ++b) {
            for (std::uint64_t i = 0; i < states[0].size(); ++i) {
                view_h(b, i) = states[b][i];
            }
        }
        Kokkos::deep_copy(_raw, view_h);
    }

    [[nodiscard]] StateVectorBatched copy() const {
        StateVectorBatched cp(_batch_size, _n_qubits);
        Kokkos::deep_copy(cp._raw, _raw);
        return cp;
    }

    std::string to_string() const {
        std::stringstream os;
        auto states_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _raw);
        Kokkos::fence();

        os << "Qubit Count : " << _n_qubits << '\n';
        os << "Dimension : " << _dim << '\n';
        for (std::uint64_t b = 0; b < _batch_size; ++b) {
            os << "--------------------\n";
            os << "Batch_id : " << b << '\n';
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
                   << " : " << states_h(b, i) << std::endl;
            }
        }
        return os.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const StateVectorBatched& states) {
        os << states.to_string();
        return os;
    }
};
}  // namespace scaluq
