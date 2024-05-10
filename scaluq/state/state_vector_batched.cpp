#include "state_vector_batched.hpp"

#include "../util/utility.hpp"

namespace scaluq {
StateVectorBatched::StateVectorBatched(UINT batch_size, UINT n_qubits)
    : _batch_size(batch_size),
      _n_qubits(n_qubits),
      _dim(1ULL << _n_qubits),
      _raw(Kokkos::View<Complex*>(Kokkos::ViewAllocateWithoutInitializing("state"),
                                  _batch_size * _dim)) {
    set_zero_state();
}

void StateVectorBatched::set_amplitude_at_index(const UINT index, const Complex& c) {
    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(const UINT i) { _raw[i * _dim + index] = c; });
}

void StateVectorBatched::set_amplitude_at_index(const UINT batch_id,
                                                const UINT index,
                                                const Complex& c) {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    host_view() = c;
    Kokkos::deep_copy(Kokkos::subview(_raw, batch_id * _dim + index), host_view());
}

std::vector<Complex> StateVectorBatched::get_amplitude_at_index(const UINT index) const {
    Kokkos::View<Complex*> res(Kokkos::ViewAllocateWithoutInitializing("res"), _batch_size);
    Kokkos::parallel_for(
        _batch_size, KOKKOS_CLASS_LAMBDA(const UINT i) { res[i] = _raw[i * _dim + index]; });
    return internal::convert_device_view_to_host_vector(res);
}

Complex StateVectorBatched::get_amplitude_at_index(const UINT batch_id, const UINT index) const {
    Kokkos::View<Complex, Kokkos::HostSpace> host_view("single_value");
    Kokkos::deep_copy(host_view, Kokkos::subview(_raw, batch_id * _dim + index));
    return host_view();
}

void StateVectorBatched::set_zero_state() {
    Kokkos::deep_copy(_raw, 0);
    set_amplitude_at_index(0, 1);
}

StateVectorBatched StateVectorBatched::Haar_random_state(UINT batch_size,
                                                         UINT n_qubits,
                                                         UINT seed) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    StateVectorBatched state(batch_size, n_qubits);
    Kokkos::parallel_for(
        state._raw.size(), KOKKOS_LAMBDA(const UINT i) {
            auto rand_gen = rand_pool.get_state();
            state._raw[i] = Complex(rand_gen.normal(0.0, 1.0), rand_gen.normal(0.0, 1.0));
            rand_pool.free_state(rand_gen);
        });
    state.normalize();
    return state;
}

std::vector<std::vector<Complex>> StateVectorBatched::amplitudes() const {
    std::vector<std::vector<Complex>> result;
    result.reserve(_batch_size);
    for (UINT batch_id = 0; batch_id < _batch_size; ++batch_id) {
        Kokkos::View<Complex*> sv =
            Kokkos::subview(_raw, std::make_pair(batch_id * _dim, (batch_id + 1) * _dim));
        result.push_back(internal::convert_device_view_to_host_vector(sv));
    }
    return result;
}

std::vector<Complex> StateVectorBatched::amplitudes(UINT batch_id) const {
    Kokkos::View<Complex*> sv =
        Kokkos::subview(_raw, std::make_pair(batch_id * _dim, (batch_id + 1) * _dim));
    return internal::convert_device_view_to_host_vector(sv);
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
                [=, *this](const UINT& i, double& lcl) {
                    UINT idx = batch_id * _dim + i;
                    lcl += squared_norm(_raw(idx));
                },
                nrm);
            norms[batch_id] = nrm;
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
                [=, *this](const UINT& i, double& lcl) {
                    UINT idx = batch_id * _dim + i;
                    lcl += squared_norm(_raw(idx));
                },
                nrm);
            nrm = Kokkos::sqrt(nrm);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, _dim), [=, *this](const UINT& i) {
                UINT idx = batch_id * _dim + i;
                _raw(idx) /= nrm;
            });
        });
}

}  // namespace scaluq
