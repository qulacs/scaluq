#pragma once

#include "../types.hpp"
#include "state_vector.hpp"

namespace scaluq {

template <std::floating_point FloatType = double, typename Space = Default>
class StateVectorBatched {
    UINT _batch_size;
    UINT _n_qubits;
    UINT _dim;
    using ExecutionSpace = ExecutionSpaceMapper<Space>::Type;
    using ComplexType = Complex<FloatType>;

public:
    Kokkos::View<ComplexType**, Kokkos::LayoutRight> _raw;
    StateVectorBatched() = default;
    StateVectorBatched(UINT batch_size, UINT n_qubits)
        : _batch_size(batch_size),
          _n_qubits(n_qubits),
          _dim(1ULL << _n_qubits),
          _raw(Kokkos::ViewAllocateWithoutInitializing("states"), _batch_size, _dim) {
        set_zero_state();
    }
    StateVectorBatched(const StateVectorBatched& other) = default;

    StateVectorBatched& operator=(const StateVectorBatched& other) = default;

    [[nodiscard]] UINT n_qubits() const { return this->_n_qubits; }

    [[nodiscard]] UINT dim() const { return this->_dim; }

    [[nodiscard]] UINT batch_size() const { return this->_batch_size; }

    void set_state_vector(const StateVector<FloatType, Space>& state) {
        if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::set_state_vector(const StateVector&): Dimensions of "
                "source and destination views do not match.");
        }
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) { _raw(batch_id, i) = state._raw(i); });
        Kokkos::fence();
    }

    template <typename TargetSpace = Space>
    [[nodiscard]] StateVector<FloatType, TargetSpace> get_state_vector(UINT batch_id) const {
        StateVector<FloatType, TargetSpace> state_vector(_n_qubits);
        Kokkos::parallel_for(
            _dim, KOKKOS_CLASS_LAMBDA(UINT i) { state_vector._raw(i) = _raw(batch_id, i); });
        Kokkos::fence();
        return state_vector;
    }

    void set_zero_state() { set_computational_basis(0); }

    void set_computational_basis(UINT basis) {
        if (basis >= _dim) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::set_computational_basis(UINT): "
                "index of "
                "computational basis must be smaller than 2^qubit_count");
        }
        Kokkos::deep_copy(_raw, 0);
        Kokkos::parallel_for(
            _batch_size, KOKKOS_CLASS_LAMBDA(UINT i) { _raw(i, basis) = 1; });
        Kokkos::fence();
    }

    void set_zero_norm_state() { Kokkos::deep_copy(_raw, 0); }

    [[nodiscard]] std::vector<std::vector<UINT>> sampling(
        UINT sampling_count, UINT seed = std::random_device()()) const {
        Kokkos::View<double**> stacked_prob("prob", _batch_size, _dim + 1);

        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                UINT batch_id = team.league_rank();
                Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, _dim),
                                      [&](UINT i, double& update, const bool final) {
                                          update += internal::squared_norm(this->_raw(batch_id, i));
                                          if (final) {
                                              stacked_prob(batch_id, i + 1) = update;
                                          }
                                      });
            });
        Kokkos::fence();

        Kokkos::View<UINT**> result(
            Kokkos::ViewAllocateWithoutInitializing("result"), _batch_size, sampling_count);
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);

        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, sampling_count}),
            KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) {
                auto rand_gen = rand_pool.get_state();
                double r = rand_gen.drand(0., 1.);
                UINT lo = 0, hi = stacked_prob.extent(1);
                while (hi - lo > 1) {
                    UINT mid = (lo + hi) / 2;
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
        std::vector vec(result.extent(0), std::vector<UINT>(result.extent(1), 0));
        for (size_t i = 0; i < result.extent(0); ++i) {
            for (size_t j = 0; j < result.extent(1); ++j) {
                vec[i][j] = result(i, j);
            }
        }
        return vec;
    }

    [[nodiscard]] static StateVectorBatched Haar_random_states(UINT batch_size,
                                                               UINT n_qubits,
                                                               bool set_same_state,
                                                               UINT seed = std::random_device()()) {
        Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
        StateVectorBatched states(batch_size, n_qubits);
        if (set_same_state) {
            states.set_state_vector(
                StateVector<FloatType, Space>::Haar_random_state(n_qubits, seed));
        } else {
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
                KOKKOS_LAMBDA(UINT b, UINT i) {
                    auto rand_gen = rand_pool.get_state();
                    states._raw(b, i) =
                        Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
                    rand_pool.free_state(rand_gen);
                });
            Kokkos::fence();
            states.normalize();
        }
        return states;
    }

    [[nodiscard]] std::vector<std::vector<ComplexType>> amplitudes() const {
        auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _raw);
        std::vector result(_raw.extent(0), std::vector<ComplexType>(_raw.extent(1), 0));
        for (size_t i = 0; i < _raw.extent(0); ++i) {
            for (size_t j = 0; j < _raw.extent(1); ++j) {
                result[i][j] = view_h(i, j);
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<double> get_squared_norm() const {
        Kokkos::View<double*> norms(Kokkos::ViewAllocateWithoutInitializing("norms"), _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                double nrm = 0;
                UINT batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [=, *this](const UINT& i, double& lcl) {
                        lcl += internal::squared_norm(_raw(batch_id, i));
                    },
                    nrm);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { norms[batch_id] = nrm; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector(norms);
    }

    void normalize() {
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                double nrm = 0;
                UINT batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [=, *this](const UINT& i, double& lcl) {
                        lcl += internal::squared_norm(_raw(batch_id, i));
                    },
                    nrm);
                team.team_barrier();
                nrm = Kokkos::sqrt(nrm);
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, _dim),
                                     [=, *this](const UINT& i) { _raw(batch_id, i) /= nrm; });
            });
        Kokkos::fence();
    }

    [[nodiscard]] std::vector<double> get_zero_probability(UINT target_qubit_index) const {
        if (target_qubit_index >= _n_qubits) {
            throw std::runtime_error(
                "Error: StateVectorBatched::get_zero_probability(UINT): index "
                "of target qubit must be smaller than qubit_count");
        }
        Kokkos::View<double*> probs("probs", _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                double sum = 0;
                UINT batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim >> 1),
                    [&](UINT i, double& lsum) {
                        UINT basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
                        lsum += internal::squared_norm(_raw(batch_id, basis_0));
                    },
                    sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
            });
        Kokkos::fence();
        return internal::convert_device_view_to_host_vector(probs);
    }
    [[nodiscard]] std::vector<double> get_marginal_probability(
        const std::vector<UINT>& measured_values) const {
        if (measured_values.size() != _n_qubits) {
            throw std::runtime_error(
                "Error: "
                "StateVectorBatched::get_marginal_probability(const vector<UINT>&): "
                "the length of measured_values must be equal to qubit_count");
        }

        std::vector<UINT> target_index;
        std::vector<UINT> target_value;
        for (UINT i = 0; i < measured_values.size(); ++i) {
            UINT measured_value = measured_values[i];
            if (measured_value == 0 || measured_value == 1) {
                target_index.push_back(i);
                target_value.push_back(measured_value);
            } else if (measured_value != StateVector<FloatType, Space>::UNMEASURED) {
                throw std::runtime_error(
                    "Error:StateVectorBatched::get_marginal_probability(const vector<UINT>&): "
                    "Invalid "
                    "qubit state specified. Each qubit state must be 0, 1, or "
                    "StateVector::UNMEASURED.");
            }
        }

        auto target_index_d = internal::convert_host_vector_to_device_view(target_index);
        auto target_value_d = internal::convert_host_vector_to_device_view(target_value);
        Kokkos::View<double*> probs("probs", _batch_size);
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                double sum = 0;
                UINT batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim >> target_index_d.size()),
                    [&](UINT i, double& lsum) {
                        UINT basis = i;
                        for (UINT cursor = 0; cursor < target_index_d.size(); cursor++) {
                            UINT insert_index = target_index_d[cursor];
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
        return internal::convert_device_view_to_host_vector(probs);
    }

    [[nodiscard]] std::vector<double> get_entropy() const {
        Kokkos::View<double*> ents("ents", _batch_size);
        const double eps = 1e-15;
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
            KOKKOS_CLASS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                    team) {
                double sum = 0;
                UINT batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, _dim),
                    [&](UINT idx, double& lsum) {
                        double prob = internal::squared_norm(_raw(batch_id, idx));
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

    void add_state_vector(const StateVectorBatched& states) {
        if (n_qubits() != states.n_qubits() || batch_size() != states.batch_size()) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::add_state_vector(const StateVectorBatched&): invalid "
                "states");
        }
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) {
                _raw(batch_id, i) += states._raw(batch_id, i);
            });
        Kokkos::fence();
    }

    void add_state_vector_with_coef(ComplexType coef, const StateVectorBatched& states) {
        if (n_qubits() != states.n_qubits() || batch_size() != states.batch_size()) [[unlikely]] {
            throw std::runtime_error(
                "Error: StateVectorBatched::add_state_vector(const StateVectorBatched&): invalid "
                "states");
        }
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) {
                _raw(batch_id, i) += coef * states._raw(batch_id, i);
            });
        Kokkos::fence();
    }

    void multiply_coef(ComplexType coef) {
        Kokkos::parallel_for(
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
            KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) { _raw(batch_id, i) *= coef; });
        Kokkos::fence();
    }

    void load(const std::vector<std::vector<ComplexType>>& states) {
        if (states.size() != _batch_size) {
            throw std::runtime_error(
                "Error: StateVectorBatched::load(std::vector<std::vector<ComplexType>>&): "
                "invalid batch_size");
        }
        for (UINT b = 0; b < states.size(); ++b) {
            if (states[b].size() != _dim) {
                throw std::runtime_error(
                    "Error: StateVectorBatched::load(std::vector<std::vector<ComplexType>>&): "
                    "invalid length of state");
            }
        }

        auto view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), _raw);
        for (UINT b = 0; b < _batch_size; ++b) {
            for (UINT i = 0; i < _dim; ++i) {
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

        os << " *** Quantum States ***\n";
        os << " * Qubit Count : " << _n_qubits << '\n';
        os << " * Dimension   : " << _dim << '\n';
        for (UINT b = 0; b < _batch_size; ++b) {
            StateVector tmp(_n_qubits);
            os << "--------------------\n";
            os << " * Batch_id    : " << b << '\n';
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
                   << ": " << states_h(b, i) << std::endl;
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
