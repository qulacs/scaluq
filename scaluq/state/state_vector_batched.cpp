#include "state_vector_batched.hpp"

#include "../util/utility.hpp"

namespace scaluq {
StateVectorBatched::StateVectorBatched(UINT batch_size, UINT n_qubits)
    : _batch_size(batch_size),
      _n_qubits(n_qubits),
      _dim(1ULL << _n_qubits),
      _raw(Kokkos::View<Complex**>(
          Kokkos::ViewAllocateWithoutInitializing("states"), _batch_size, _dim)) {
    set_zero_state();
}

void StateVectorBatched::set_state_vector_at_batch_id(UINT batch_id, StateVector states) {
    if (_raw.extent(1) != states._raw.extent(0)) {
        throw std::runtime_error("Error: Dimensions of source and destination views do not match.");
    }
    auto sv = Kokkos::subview(_raw, batch_id, Kokkos::ALL());
    Kokkos::deep_copy(sv, states._raw);
}

StateVector StateVectorBatched::get_state_vector_at_batch_id(UINT batch_id) const {
    StateVectorView ret("ret", _dim);
    StateVectorView sv = Kokkos::subview(_raw, batch_id, Kokkos::ALL());
    Kokkos::deep_copy(ret, sv);
    return ret;
}

void StateVectorBatched::set_amplitude_at_index(UINT index, const Complex& c) {
    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(UINT b) { _raw(b, index) = c; });
}

std::vector<Complex> StateVectorBatched::get_amplitude_at_index(UINT index) const {
    StateVectorView res(Kokkos::ViewAllocateWithoutInitializing("res"), _batch_size);

    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(UINT b) { res[b] = _raw(b, index); });
    return internal::convert_device_view_to_host_vector(res);
}

void StateVectorBatched::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(UINT i) { _raw(i, 0) = 1; });
}

StateVectorBatched StateVectorBatched::Haar_random_states(UINT batch_size,
                                                          UINT n_qubits,
                                                          UINT seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVectorBatched states(batch_size, n_qubits);
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {states.batch_size(), states.dim()}),
        KOKKOS_LAMBDA(UINT b, UINT i) {
            auto rand_gen = rand_pool.get_state();
            states._raw(b, i) = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    states.normalize();
    return states;
}

std::vector<std::vector<Complex>> StateVectorBatched::amplitudes() const {
    std::vector<std::vector<Complex>> result;
    result.reserve(_batch_size);
    for (UINT batch_id = 0; batch_id < _batch_size; ++batch_id) {
        result.push_back(get_state_vector_at_batch_id(batch_id).amplitudes());
    }
    return result;
}

std::vector<double> StateVectorBatched::get_squared_norm() const {
    Kokkos::View<double*> norms(Kokkos::ViewAllocateWithoutInitializing("norms"), _batch_size);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                team) {
            double nrm = 0;
            UINT batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, _dim),
                [=, *this](const UINT& i, double& lcl) { lcl += squared_norm(_raw(batch_id, i)); },
                nrm);
            Kokkos::single(Kokkos::PerTeam(team), [&] { norms[batch_id] = nrm; });
            ;
        });
    return internal::convert_device_view_to_host_vector(norms);
}

void StateVectorBatched::normalize() {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                team) {
            double nrm = 0;
            UINT batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, _dim),
                [=, *this](const UINT& i, double& lcl) { lcl += squared_norm(_raw(batch_id, i)); },
                nrm);
            nrm = Kokkos::sqrt(nrm);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, _dim),
                                 [=, *this](const UINT& i) { _raw(batch_id, i) /= nrm; });
        });
}

std::vector<double> StateVectorBatched::get_zero_probability(UINT target_qubit_index) const {
    if (target_qubit_index >= _n_qubits) {
        throw std::runtime_error(
            "Error: StateVectorBatched::get_zero_probability(UINT): index "
            "of target qubit must be smaller than qubit_count");
    }
    Kokkos::View<double*> probs("d_prob", _batch_size);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                team) {
            double sum;
            UINT batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                "zero_prob",
                _dim >> 1,
                [=, *this](const UINT& i, double& lsum) {
                    UINT basis_0 = internal::insert_zero_to_basis_index(i, target_qubit_index);
                    lsum += squared_norm(this->_raw(batch_id, basis_0));
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&] { probs[batch_id] = sum; });
        });
    return internal::convert_device_view_to_host_vector(probs);
}

std::vector<double> StateVectorBatched::get_marginal_probability(
    const std::vector<UINT>& measured_values) const {
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
        } else if (measured_value != StateVector::UNMEASURED) {
            throw std::runtime_error(
                "Error: Invalid qubit state specified. Each qubit state must be 0, 1, or "
                "StateVector::UNMEASURED.");
        }
    }

    auto d_target_index = internal::convert_host_vector_to_device_view(target_index);
    auto d_target_value = internal::convert_host_vector_to_device_view(target_value);
    Kokkos::View<double*> probs("d_prob", _batch_size);
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                team) {
            double sum;
            UINT batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                _dim >> target_index.size(),
                [=, *this](const UINT& i, double& lsum) {
                    UINT basis = i;
                    for (UINT cursor = 0; cursor < d_target_index.size(); cursor++) {
                        UINT insert_index = d_target_index[cursor];
                        basis = internal::insert_zero_to_basis_index(basis, insert_index);
                        basis ^= d_target_value[cursor] << insert_index;
                    }
                    lsum += squared_norm(this->_raw(batch_id, basis));
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [=] { probs[batch_id] = sum; });
        });
    return internal::convert_device_view_to_host_vector(probs);
}

std::vector<double> StateVectorBatched::get_entropy() const {
    Kokkos::View<double*> ents("d_ents", _batch_size);
    const double eps = 1e-15;
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<>(_batch_size, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::TeamPolicy::member_type&
                team) {
            double sum;
            UINT batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                "get_entropy",
                _dim,
                [=, *this](const UINT& i, double& lsum) {
                    double prob = squared_norm(_raw(batch_id, i));
                    prob = Kokkos::max(prob, eps);
                    lsum += -prob * Kokkos::log(prob);
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [=] { ents[batch_id] = sum; });
        });
    return internal::convert_device_view_to_host_vector(ents);
}

void StateVectorBatched::add_state_vector(const StateVectorBatched& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) {
            _raw(batch_id, i) += states._raw(batch_id, i);
        });
}
void StateVectorBatched::add_state_vector_with_coef(const Complex& coef,
                                                    const StateVectorBatched& states) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) {
            _raw(batch_id, i) += coef * states._raw(batch_id, i);
        });
}
void StateVectorBatched::multiply_coef(const Complex& coef) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) { _raw(batch_id, i) *= coef; });
}

std::string StateVectorBatched::to_string() const {
    std::stringstream os;

    os << " *** Quantum States ***\n";
    os << " * Qubit Count : " << _n_qubits << '\n';
    os << " * Dimension   : " << _dim << '\n';
    for (UINT b = 0; b < _batch_size; ++b) {
        StateVector tmp = StateVectorView(Kokkos::subview(_raw, b, Kokkos::ALL()));
        auto amp = tmp.amplitudes();
        os << "--------------------\n";
        os << " * Batch_id    : " << b << '\n';
        os << " * State vector : \n";
        for (UINT i = 0; i < _dim; ++i) {
            os << amp[i] << std::endl;
        }
    }
    return os.str();
}

void StateVectorBatched::load(const std::vector<std::vector<Complex>>& other) {
    if (other.size() != _batch_size) {
        throw std::runtime_error(
            "Error: StateVectorBatched::load(std::vector<std::vector<Complex>>&): invalid "
            "batch_size");
    }
    for (UINT b = 0; b < other.size(); ++b) {
        if (other[b].size() != _dim) {
            throw std::runtime_error(
                "Error: StateVectorBatched::load(std::vector<std::vector<Complex>>&): invalid "
                "length of state");
        }
    }

    for (UINT b = 0; b < other.size(); ++b) {
        auto device_view = internal::convert_host_vector_to_device_view<Complex>(other[b]);
        auto sv = Kokkos::subview(_raw, b, Kokkos::ALL());
        Kokkos::deep_copy(sv, device_view);
    }
}

}  // namespace scaluq
