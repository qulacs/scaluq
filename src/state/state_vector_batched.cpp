#include <scaluq/state/state_vector_batched.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
StateVectorBatched<Prec, Space>::StateVectorBatched(std::uint64_t batch_size,
                                                    std::uint64_t n_qubits)
    : _batch_size(batch_size),
      _n_qubits(n_qubits),
      _dim(1ULL << _n_qubits),
      _raw(Kokkos::ViewAllocateWithoutInitializing("states"), _batch_size, _dim) {
    set_zero_state();
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::set_state_vector(const StateVector<Prec, Space>& state) {
    if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
        throw std::runtime_error(
            "Error: StateVectorBatched::set_state_vector(const StateVector&): Dimensions of "
            "source and destination views do not match.");
    }
    Kokkos::parallel_for(
        "set_state_vector",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            _raw(batch_id, i) = state._raw(i);
        });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::set_state_vector_at(std::uint64_t batch_id,
                                                          const StateVector<Prec, Space>& state) {
    if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
        throw std::runtime_error(
            "Error: StateVectorBatched::set_state_vector(std::uint64_t, const StateVector&): "
            "Dimensions of source and destination views do not match.");
    }
    Kokkos::parallel_for(
        "set_state_vector_at",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, _dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _raw(batch_id, i) = state._raw(i); });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
StateVector<Prec, Space> StateVectorBatched<Prec, Space>::get_state_vector_at(
    std::uint64_t batch_id) const {
    auto ret = StateVector<Prec, Space>::uninitialized_state(_n_qubits);
    Kokkos::parallel_for(
        "get_state_vector_at",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, _dim),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { ret._raw(i) = _raw(batch_id, i); });
    Kokkos::fence();
    return ret;
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::set_computational_basis(std::uint64_t basis) {
    if (basis >= _dim) [[unlikely]] {
        throw std::runtime_error(
            "Error: StateVectorBatched::set_computational_basis(std::uint64_t): "
            "index of computational basis must be smaller than 2^qubit_count");
    }
    Kokkos::parallel_for(
        "set_computational_basis",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t b, std::uint64_t i) { _raw(b, i) = (i == basis); });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::set_zero_norm_state() {
    Kokkos::deep_copy(_raw, 0.);
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::set_Haar_random_state(bool set_same_state,
                                                            std::uint64_t seed) {
    *this = Haar_random_state(_batch_size, _n_qubits, set_same_state, seed);
}

template <Precision Prec, ExecutionSpace Space>
std::vector<std::vector<std::uint64_t>> StateVectorBatched<Prec, Space>::sampling(
    std::uint64_t sampling_count, std::uint64_t seed) const {
    Kokkos::View<FloatType**, internal::SpaceType<Space>> stacked_prob(
        "prob", _batch_size, _dim + 1);

    Kokkos::parallel_for(
        "sampling (compute stacked prob)",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
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

    std::vector result(_batch_size, std::vector<std::uint64_t>(sampling_count));
    Kokkos::Random_XorShift64_Pool<internal::SpaceType<Space>> rand_pool(seed);
    std::vector<std::uint64_t> batch_todo(_batch_size * sampling_count);
    std::vector<std::uint64_t> sample_todo(_batch_size * sampling_count);
    for (std::uint64_t i = 0; i < _batch_size; i++) {
        for (std::uint64_t j = 0; j < sampling_count; j++) {
            std::uint64_t idx = i * sampling_count + j;
            batch_todo[idx] = i;
            sample_todo[idx] = j;
        }
    }
    while (!batch_todo.empty()) {
        std::size_t todo_count = batch_todo.size();
        Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> batch_ids =
            internal::convert_vector_to_view<std::uint64_t, Space>(batch_todo);
        Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> result_buf(
            Kokkos::ViewAllocateWithoutInitializing("result_buf"), todo_count);
        Kokkos::parallel_for(
            "sampling (choose)",
            Kokkos::RangePolicy<internal::SpaceType<Space>>(0, todo_count),
            KOKKOS_CLASS_LAMBDA(std::uint64_t idx) {
                std::uint64_t batch_id = batch_ids[idx];
                auto rand_gen = rand_pool.get_state();
                FloatType r = static_cast<FloatType>(rand_gen.drand(0., 1.));
                std::uint64_t lo = 0, hi = stacked_prob.extent(1);
                while (hi - lo > 1) {
                    std::uint64_t mid = (lo + hi) / 2;
                    if (stacked_prob(batch_id, mid) > r) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                result_buf(idx) = lo;
                rand_pool.free_state(rand_gen);
            });
        Kokkos::fence();
        auto result_buf_host = internal::convert_view_to_vector<std::uint64_t, Space>(result_buf);
        // Especially for F16 and BF16, sampling sometimes fails with result == _dim.
        // In this case, re-sampling is performed.
        std::vector<std::uint64_t> next_batch_todo;
        std::vector<std::uint64_t> next_sample_todo;
        for (std::size_t i = 0; i < todo_count; i++) {
            if (result_buf_host[i] == _dim) {
                next_batch_todo.push_back(batch_todo[i]);
                next_sample_todo.push_back(sample_todo[i]);
            } else {
                result[batch_todo[i]][sample_todo[i]] = result_buf_host[i];
            }
        }
        batch_todo.swap(next_batch_todo);
        sample_todo.swap(next_sample_todo);
    }
    return result;
}

template <Precision Prec, ExecutionSpace Space>
StateVectorBatched<Prec, Space> StateVectorBatched<Prec, Space>::Haar_random_state(
    std::uint64_t batch_size, std::uint64_t n_qubits, bool set_same_state, std::uint64_t seed) {
    Kokkos::Random_XorShift64_Pool<internal::SpaceType<Space>> rand_pool(seed);
    auto states(StateVectorBatched<Prec, Space>::uninitialized_state(batch_size, n_qubits));
    if (set_same_state) {
        states.set_state_vector(StateVector<Prec, Space>::Haar_random_state(n_qubits, seed));
    } else {
        Kokkos::parallel_for(
            "Haar_random_state",
            Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>(
                {0, 0}, {states.batch_size(), states.dim()}),
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

template <Precision Prec, ExecutionSpace Space>
StateVectorBatched<Prec, Space> StateVectorBatched<Prec, Space>::uninitialized_state(
    std::uint64_t batch_size, std::uint64_t n_qubits) {
    StateVectorBatched states;
    states._n_qubits = n_qubits;
    states._dim = 1ULL << n_qubits;
    states._batch_size = batch_size;
    states._raw = Kokkos::View<ComplexType**, Kokkos::LayoutRight, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("states"), states._batch_size, states._dim);
    return states;
}

template <Precision Prec, ExecutionSpace Space>
std::vector<std::vector<StdComplex>> StateVectorBatched<Prec, Space>::get_amplitudes() const {
    auto view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), _raw);
    std::vector vv(_raw.extent(0), std::vector(_raw.extent(1), StdComplex(0.)));
    for (size_t i = 0; i < view_h.extent(0); ++i) {
        for (size_t j = 0; j < view_h.extent(1); ++j) {
            vv[i][j] = static_cast<StdComplex>(view_h(i, j));
        }
    }
    return vv;
}

template <Precision Prec, ExecutionSpace Space>
std::vector<double> StateVectorBatched<Prec, Space>::get_squared_norm() const {
    Kokkos::View<FloatType*, internal::SpaceType<Space>> norms(
        Kokkos::ViewAllocateWithoutInitializing("norms"), _batch_size);
    Kokkos::parallel_for(
        "get_squared_norm",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
            FloatType nrm = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, _dim),
                [&](const std::uint64_t& i, FloatType& lcl) {
                    lcl += internal::squared_norm(_raw(batch_id, i));
                },
                internal::Sum<FloatType, Space>(nrm));
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] { norms[batch_id] = nrm; });
        });
    Kokkos::fence();
    std::vector<FloatType> norms_host_prec =
        internal::convert_view_to_vector<FloatType, Space>(norms);
    std::vector<double> norms_double(_batch_size);
    std::ranges::copy(norms_host_prec, norms_double.begin());
    return norms_double;
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::normalize() {
    Kokkos::parallel_for(
        "normalize",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
            FloatType nrm = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, _dim),
                [&](const std::uint64_t& i, FloatType& lcl) {
                    lcl += internal::squared_norm(_raw(batch_id, i));
                },
                internal::Sum<FloatType, Space>(nrm));
            team.team_barrier();
            nrm = internal::sqrt(nrm);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, _dim),
                                 [&](const std::uint64_t& i) { _raw(batch_id, i) /= nrm; });
        });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
std::vector<double> StateVectorBatched<Prec, Space>::get_zero_probability(
    std::uint64_t target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVectorBatched::get_zero_probability(std::uint64_t): index "
            "of target qubit must be smaller than qubit_count");
    }
    Kokkos::View<FloatType*, internal::SpaceType<Space>> probs("probs", _batch_size);
    Kokkos::parallel_for(
        "get_zero_probability",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
            FloatType sum = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, _dim >> 1),
                [&](std::uint64_t i, FloatType& lsum) {
                    std::uint64_t basis_0 =
                        internal::insert_zero_to_basis_index(i, target_qubit_index);
                    lsum += internal::squared_norm(_raw(batch_id, basis_0));
                },
                internal::Sum<FloatType, Space>(sum));
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
        });
    Kokkos::fence();
    std::vector<FloatType> probs_host_prec =
        internal::convert_view_to_vector<FloatType, Space>(probs);
    std::vector<double> probs_double(_batch_size);
    std::ranges::copy(probs_host_prec, probs_double.begin());
    return probs_double;
}

template <Precision Prec, ExecutionSpace Space>
std::vector<double> StateVectorBatched<Prec, Space>::get_marginal_probability(
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
        } else if (measured_value != StateVector<Prec, Space>::UNMEASURED) {
            throw std::runtime_error(
                "Error:StateVectorBatched::get_marginal_probability(const "
                "vector<std::uint64_t>&): Invalid qubit state specified. Each qubit state must "
                "be 0, 1, or StateVector::UNMEASURED.");
        }
    }

    auto target_index_d = internal::convert_vector_to_view<std::uint64_t, Space>(target_index);
    auto target_value_d = internal::convert_vector_to_view<std::uint64_t, Space>(target_value);
    Kokkos::View<FloatType*, internal::SpaceType<Space>> probs("probs", _batch_size);
    Kokkos::parallel_for(
        "get_marginal_probability",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
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
                internal::Sum<FloatType, Space>(sum));
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
        });
    Kokkos::fence();
    std::vector<FloatType> probs_host_prec =
        internal::convert_view_to_vector<FloatType, Space>(probs);
    std::vector<double> probs_double(_batch_size);
    std::ranges::copy(probs_host_prec, probs_double.begin());
    return probs_double;
}

template <Precision Prec, ExecutionSpace Space>
std::vector<double> StateVectorBatched<Prec, Space>::get_entropy() const {
    Kokkos::View<FloatType*, internal::SpaceType<Space>> ents("ents", _batch_size);
    const FloatType eps = static_cast<FloatType>(1e-15);
    Kokkos::parallel_for(
        "get_entropy",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), _batch_size, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<internal::SpaceType<Space>>::TeamPolicy::member_type& team) {
            FloatType sum = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, _dim),
                [&](std::uint64_t idx, FloatType& lsum) {
                    FloatType prob = internal::squared_norm(_raw(batch_id, idx));
                    prob = Kokkos::max(prob, eps);
                    lsum += -prob * internal::log2(prob);
                },
                internal::Sum<FloatType, Space>(sum));
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] { ents[batch_id] = sum; });
        });
    Kokkos::fence();
    std::vector<FloatType> ents_host_prec =
        internal::convert_view_to_vector<FloatType, Space>(ents);
    std::vector<double> ents_double(_batch_size);
    std::ranges::copy(ents_host_prec, ents_double.begin());
    return ents_double;
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::add_state_vector_with_coef(const StdComplex& coef,
                                                                 const StateVectorBatched& states) {
    if (n_qubits() != states.n_qubits() || batch_size() != states.batch_size()) [[unlikely]] {
        throw std::runtime_error(
            "Error: StateVectorBatched::add_state_vector(const StateVectorBatched&): invalid "
            "states");
    }
    Kokkos::parallel_for(
        "add_state_vector_with_coef",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            _raw(batch_id, i) += ComplexType(coef) * states._raw(batch_id, i);
        });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::multiply_coef(const StdComplex& coef) {
    Kokkos::parallel_for(
        "multiply_coef",
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(std::uint64_t batch_id, std::uint64_t i) {
            _raw(batch_id, i) *= ComplexType(coef);
        });
    Kokkos::fence();
}

template <Precision Prec, ExecutionSpace Space>
void StateVectorBatched<Prec, Space>::load(const std::vector<std::vector<StdComplex>>& states) {
    if (states.size() != _batch_size) {
        throw std::runtime_error(
            "Error: StateVectorBatched::load(std::vector<std::vector<Complex<Prec>>>&): "
            "invalid batch_size");
    }
    for (std::uint64_t b = 0; b < states.size(); ++b) {
        if (states[b].size() != _dim) {
            throw std::runtime_error(
                "Error: "
                "StateVectorBatched::load(std::vector<std::vector<Complex<Prec>>>&): "
                "invalid length of state");
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

template <Precision Prec, ExecutionSpace Space>
StateVectorBatched<Prec, Space> StateVectorBatched<Prec, Space>::copy() const {
    auto cp = StateVectorBatched<Prec, Space>::uninitialized_state(_batch_size, _n_qubits);
    Kokkos::deep_copy(cp._raw, _raw);
    return cp;
}

template <Precision Prec, ExecutionSpace Space>
std::string StateVectorBatched<Prec, Space>::to_string() const {
    std::stringstream os;
    auto amp = this->get_amplitudes();
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
               << " : " << amp[b][i] << std::endl;
        }
    }
    return os.str();
}

template class StateVectorBatched<internal::Prec, internal::Space>;

}  // namespace scaluq
