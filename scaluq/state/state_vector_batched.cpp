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

void StateVectorBatched::set_state_vector(const StateVector& state) {
    if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
        throw std::runtime_error("Error: Dimensions of source and destination views do not match.");
    }
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) { _raw(batch_id, i) = state._raw(i); });
    Kokkos::fence();
}

void StateVectorBatched::set_state_vector(UINT batch_id, const StateVector& state) {
    if (_raw.extent(1) != state._raw.extent(0)) [[unlikely]] {
        throw std::runtime_error("Error: Dimensions of source and destination views do not match.");
    }
    Kokkos::parallel_for(
        _dim, KOKKOS_CLASS_LAMBDA(UINT i) { _raw(batch_id, i) = state._raw(i); });
    Kokkos::fence();
}

StateVector StateVectorBatched::get_state_vector(UINT batch_id) const {
    StateVector ret(_n_qubits);
    Kokkos::parallel_for(
        _dim, KOKKOS_CLASS_LAMBDA(UINT i) { ret._raw(i) = _raw(batch_id, i); });
    Kokkos::fence();
    return ret;
}

void StateVectorBatched::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(UINT i) { _raw(i, 0) = 1; });
    Kokkos::fence();
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
    Kokkos::fence();
    states.normalize();
    return states;
}

std::vector<std::vector<Complex>> StateVectorBatched::amplitudes() const {
    std::vector<std::vector<Complex>> result;
    result.reserve(_batch_size);
    for (UINT batch_id = 0; batch_id < _batch_size; ++batch_id) {
        result.push_back(get_state_vector(batch_id).amplitudes());
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
        });
    Kokkos::fence();
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
    Kokkos::fence();
}

void StateVectorBatched::add_state_vector(const StateVectorBatched& states) {
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
void StateVectorBatched::add_state_vector_with_coef(const Complex& coef,
                                                    const StateVectorBatched& states) {
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
void StateVectorBatched::multiply_coef(const Complex& coef) {
    Kokkos::parallel_for(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {_batch_size, _dim}),
        KOKKOS_CLASS_LAMBDA(UINT batch_id, UINT i) { _raw(batch_id, i) *= coef; });
    Kokkos::fence();
}

void StateVectorBatched::load(const std::vector<std::vector<Complex>>& states) {
    if (states.size() != _batch_size) {
        throw std::runtime_error(
            "Error: StateVectorBatched::load(std::vector<std::vector<Complex>>&): invalid "
            "batch_size");
    }
    for (UINT b = 0; b < states.size(); ++b) {
        if (states[b].size() != _dim) {
            throw std::runtime_error(
                "Error: StateVectorBatched::load(std::vector<std::vector<Complex>>&): invalid "
                "length of state");
        }
    }

    auto view_h = Kokkos::create_mirror_view(Kokkos::HostSpace(), _raw);
    Kokkos::fence();
    for (UINT b = 0; b < states.size(); ++b) {
        for (UINT i = 0; i < states[0].size(); ++i) {
            view_h(b, i) = states[b][i];
        }
    }
    Kokkos::deep_copy(_raw, view_h);
    Kokkos::fence();
}

StateVectorBatched StateVectorBatched::copy() const {
    StateVectorBatched cp(_batch_size, _n_qubits);
    Kokkos::deep_copy(cp._raw, _raw);
    Kokkos::fence();
    return cp;
}

}  // namespace scaluq
