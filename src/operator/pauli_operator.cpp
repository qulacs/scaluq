#include <bitset>
#include <scaluq/operator/pauli_operator.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"
#include "apply_pauli.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space>::PauliOperator(std::uint64_t n_qubits,
                                          std::string_view pauli_string,
                                          StdComplex coef)
    : _n_qubits(n_qubits), _coef(coef) {
    auto ss = std::stringstream(std::string(pauli_string));
    while (1) {
        char pauli;
        std::uint64_t target;
        ss >> pauli;
        if (ss.fail()) break;
        ss >> target;
        if (ss.fail()) {
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }
        if (target >= n_qubits) {
            throw std::runtime_error(
                "PauliOperator::PauliOperator: target qubit index is larger than n_qubits");
        }
        std::uint64_t pauli_id = [&] {
            if (pauli == 'I' || pauli == 'i') return PauliOperator::I;
            if (pauli == 'X' || pauli == 'x') return PauliOperator::X;
            if (pauli == 'Y' || pauli == 'y') return PauliOperator::Y;
            if (pauli == 'Z' || pauli == 'z') return PauliOperator::Z;
            throw std::runtime_error("PauliOperator::PauliOperator: invalid pauli_string format");
        }();
        if (pauli_id != 0) add_single_pauli(target, pauli_id);
    }
}

template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space>::PauliOperator(std::uint64_t n_qubits,
                                          const std::vector<std::uint64_t>& target_qubit_list,
                                          const std::vector<std::uint64_t>& pauli_id_list,
                                          StdComplex coef)
    : _n_qubits(n_qubits), _coef(coef) {
    if (target_qubit_list.size() != pauli_id_list.size()) {
        throw std::runtime_error(
            "PauliOperator::PauliOperator: target_qubit_list must have same size to "
            "pauli_id_list");
    }
    for (std::uint64_t term_index = 0; term_index < target_qubit_list.size(); ++term_index) {
        if (target_qubit_list[term_index] >= n_qubits) {
            throw std::runtime_error(
                "PauliOperator::PauliOperator: target qubit index is larger than n_qubits");
        }
        if (pauli_id_list[term_index] != 0) {
            add_single_pauli(target_qubit_list[term_index], pauli_id_list[term_index]);
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space>::PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit,
                                          StdComplex coef)
    : _n_qubits(pauli_id_par_qubit.size()), _coef(coef) {
    for (std::uint64_t i = 0; i < pauli_id_par_qubit.size(); ++i) {
        if (pauli_id_par_qubit[i] != PauliOperator<Prec, Space>::I) {
            add_single_pauli(i, pauli_id_par_qubit[i]);
        }
    }
}

template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space>::PauliOperator(std::uint64_t n_qubits,
                                          std::uint64_t bit_flip_mask,
                                          std::uint64_t phase_flip_mask,
                                          StdComplex coef)
    : _n_qubits(n_qubits),
      _coef(coef),
      _bit_flip_mask(bit_flip_mask),
      _phase_flip_mask(phase_flip_mask) {
    if (n_qubits != 64 && (bit_flip_mask | phase_flip_mask) >> n_qubits) {
        throw std::runtime_error(
            "PauliOperator::PauliOperator: operand mask is larger than n_qubits");
    }
}

template <Precision Prec, ExecutionSpace Space>
void PauliOperator<Prec, Space>::add_single_pauli(std::uint64_t target_qubit,
                                                  std::uint64_t pauli_id) {
    if (target_qubit >= sizeof(std::uint64_t) * 8) {
        throw std::runtime_error("PauliOperator::add_single_pauli: target_qubit is too large");
    }
    if (target_qubit >= _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::add_single_pauli: target_qubit is larger than n_qubits");
    }
    if (pauli_id >= 4) {
        throw std::runtime_error("PauliOperator::add_single_pauli: pauli_id is invalid");
    }
    if ((_bit_flip_mask | _phase_flip_mask) >> target_qubit & 1) {
        throw std::runtime_error(
            "PauliOperator::add_single_pauli: You cannot add single pauli twice for "
            "same qubit.");
    }
    if (pauli_id == PauliOperator::X || pauli_id == PauliOperator::Y) {
        _bit_flip_mask |= 1ULL << target_qubit;
    }
    if (pauli_id == PauliOperator::Y || pauli_id == PauliOperator::Z) {
        _phase_flip_mask |= 1ULL << target_qubit;
    }
}

template <Precision Prec, ExecutionSpace Space>
const std::vector<std::uint64_t> PauliOperator<Prec, Space>::target_qubit_list() const {
    return internal::mask_to_vector(_bit_flip_mask | _phase_flip_mask);
}
template <Precision Prec, ExecutionSpace Space>
const std::vector<std::uint64_t> PauliOperator<Prec, Space>::pauli_id_list() const {
    std::vector<std::uint64_t> id_list;
    for (std::uint64_t sub_mask = _bit_flip_mask | _phase_flip_mask; sub_mask;
         sub_mask &= (sub_mask - 1)) {
        std::uint64_t q = std::countr_zero(sub_mask);
        if ((_bit_flip_mask & _phase_flip_mask) >> q & 1) {
            id_list.push_back(PauliID::Y);
        } else if (_bit_flip_mask >> q & 1) {
            id_list.push_back(PauliID::X);
        } else if (_phase_flip_mask >> q & 1) {
            id_list.push_back(PauliID::Z);
        }
    }
    return id_list;
}

template <Precision Prec, ExecutionSpace Space>
std::string PauliOperator<Prec, Space>::get_pauli_string() const {
    auto target_qubit_list = this->target_qubit_list();
    auto pauli_id_list = this->pauli_id_list();
    std::string res;
    for (std::uint64_t i = 0; i < target_qubit_list.size(); ++i) {
        std::uint64_t target_qubit = target_qubit_list[i];
        std::uint64_t pauli_id = pauli_id_list[i];
        res += [&] {
            switch (pauli_id) {
                case PauliOperator::I:
                    return "I ";
                case PauliOperator::X:
                    return "X ";
                case PauliOperator::Y:
                    return "Y ";
                case PauliOperator::Z:
                    return "Z ";
                default:
                    throw std::runtime_error(
                        "PauliOperator::get_pauli_string: Invalid Pauli ID encountered");
            }
        }();
        res += std::to_string(target_qubit);
        if (i + 1 != target_qubit_list.size()) res += ' ';
    }
    return res;
}

template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space> PauliOperator<Prec, Space>::get_dagger() const {
    return PauliOperator(
        _n_qubits, _bit_flip_mask, _phase_flip_mask, scaluq::internal::conj(_coef));
}

template <Precision Prec, ExecutionSpace Space>
void PauliOperator<Prec, Space>::apply_to_state(StateVector<Prec, Space>& state_vector) const {
    if (state_vector.n_qubits() != _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::apply_to_state: n_qubits of state_vector is  not equal to n_qubits "
            "of the operator");
    }
    internal::apply_pauli(0ULL, 0LL, _bit_flip_mask, _phase_flip_mask, _coef, state_vector);
}

template <Precision Prec, ExecutionSpace Space>
StdComplex PauliOperator<Prec, Space>::get_expectation_value(
    const StateVector<Prec, Space>& state_vector) const {
    if (state_vector.n_qubits() != _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::get_expectation_value: n_qubits of state_vector is not equal to "
            "n_qubits of the operator");
    }
    std::uint64_t bit_flip_mask = _bit_flip_mask;
    std::uint64_t phase_flip_mask = _phase_flip_mask;
    if (bit_flip_mask == 0) {
        FloatType res;
        Kokkos::parallel_reduce(
            "get_expectation_value",
            Kokkos::RangePolicy<internal::SpaceType<Space>>(0, state_vector.dim()),
            KOKKOS_LAMBDA(std::uint64_t state_idx, FloatType & sum) {
                FloatType tmp = (scaluq::internal::conj(state_vector._raw[state_idx]) *
                                 state_vector._raw[state_idx])
                                    .real();
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            internal::Sum<FloatType, Space>(res));
        return _coef * res;
    }
    std::uint64_t pivot = std::bit_width(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    ComplexType global_phase = internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
    FloatType res;
    Kokkos::parallel_reduce(
        "get_expectation_value",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, state_vector.dim() >> 1),
        KOKKOS_LAMBDA(std::uint64_t state_idx, FloatType & sum) {
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            FloatType tmp = scaluq::internal::real(
                state_vector._raw[basis_0] * scaluq::internal::conj(state_vector._raw[basis_1]) *
                global_phase * FloatType{2});
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
            sum += tmp;
        },
        internal::Sum<FloatType, Space>(res));
    return static_cast<StdComplex>(_coef * res);
}

template <Precision Prec, ExecutionSpace Space>
std::vector<StdComplex> PauliOperator<Prec, Space>::get_expectation_value(
    const StateVectorBatched<Prec, Space>& states) const {
    if (states.n_qubits() != _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::get_expectation_value: n_qubits of states is not equal to n_qubits "
            "of the operator");
    }
    std::uint64_t bit_flip_mask = _bit_flip_mask;
    std::uint64_t phase_flip_mask = _phase_flip_mask;
    if (bit_flip_mask == 0) {
        Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<Space>> results(
            "results", states.batch_size());
        Kokkos::parallel_for(
            "get_expectation_value",
            Kokkos::TeamPolicy<internal::SpaceType<Space>>(
                internal::SpaceType<Space>(), states.batch_size(), Kokkos::AUTO),
            KOKKOS_CLASS_LAMBDA(
                const typename Kokkos::TeamPolicy<internal::SpaceType<Space>>::member_type& team) {
                FloatType res = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, states.dim()),
                    [&](std::uint64_t state_idx, FloatType& sum) {
                        FloatType tmp = (scaluq::internal::conj(states._raw(batch_id, state_idx)) *
                                         states._raw(batch_id, state_idx))
                                            .real();
                        if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                        sum += tmp;
                    },
                    internal::Sum<FloatType, Space>(res));
                ComplexType cres = _coef * res;
                results(batch_id) = Kokkos::complex<double>(cres.real(), cres.imag());
            });
        Kokkos::fence();
        auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
        return std::vector<StdComplex>(results_h.data(), results_h.data() + results_h.size());
    }

    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<Space>> results("results",
                                                                               states.batch_size());
    std::uint64_t pivot = std::bit_width(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    ComplexType global_phase = internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
    Kokkos::parallel_for(
        "get_expectation_value",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), states.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<Space>>::member_type& team) {
            FloatType res = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, states.dim()),
                [&](std::uint64_t state_idx, FloatType& sum) {
                    std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                    std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                    FloatType tmp = scaluq::internal::real(
                        states._raw(batch_id, basis_0) *
                        scaluq::internal::conj(states._raw(batch_id, basis_1)) * global_phase *
                        FloatType{2});
                    if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                    sum += tmp;
                },
                internal::Sum<FloatType, Space>(res));
            ComplexType cres = _coef * res;
            results(batch_id) = Kokkos::complex<double>(cres.real(), cres.imag());
        });
    Kokkos::fence();
    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    return std::vector<StdComplex>(results_h.data(), results_h.data() + results_h.size());
}

template <Precision Prec, ExecutionSpace Space>
StdComplex PauliOperator<Prec, Space>::get_transition_amplitude(
    const StateVector<Prec, Space>& state_vector_bra,
    const StateVector<Prec, Space>& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error("state_vector_bra must have same n_qubits to state_vector_ket.");
    }
    if (state_vector_bra.n_qubits() != _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::get_transition_amplitude: n_qubits of state_vector is too small to "
            "apply the operator");
    }
    std::uint64_t bit_flip_mask = _bit_flip_mask;
    std::uint64_t phase_flip_mask = _phase_flip_mask;
    if (bit_flip_mask == 0) {
        ComplexType res;
        Kokkos::parallel_reduce(
            "get_transition_amplitude",
            Kokkos::RangePolicy<internal::SpaceType<Space>>(0, state_vector_bra.dim()),
            KOKKOS_LAMBDA(std::uint64_t state_idx, ComplexType & sum) {
                ComplexType tmp = scaluq::internal::conj(state_vector_bra._raw[state_idx]) *
                                  state_vector_ket._raw[state_idx];
                if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                sum += tmp;
            },
            internal::Sum<ComplexType, Space>(res));
        Kokkos::fence();
        return _coef * res;
    }
    std::uint64_t pivot = std::bit_width(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    ComplexType global_phase = internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
    ComplexType res;
    Kokkos::parallel_reduce(
        "get_transition_amplitude",
        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, state_vector_bra.dim() >> 1),
        KOKKOS_LAMBDA(std::uint64_t state_idx, ComplexType & sum) {
            std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
            std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
            ComplexType tmp1 = scaluq::internal::conj(state_vector_bra._raw[basis_1]) *
                               state_vector_ket._raw[basis_0] * global_phase;
            if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
            ComplexType tmp2 = scaluq::internal::conj(state_vector_bra._raw[basis_0]) *
                               state_vector_ket._raw[basis_1] * global_phase;
            if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
            sum += tmp1 + tmp2;
        },
        internal::Sum<ComplexType, Space>(res));
    Kokkos::fence();
    return static_cast<StdComplex>(_coef * res);
}

template <Precision Prec, ExecutionSpace Space>
std::vector<StdComplex> PauliOperator<Prec, Space>::get_transition_amplitude(
    const StateVectorBatched<Prec, Space>& states_bra,
    const StateVectorBatched<Prec, Space>& states_ket) const {
    if (states_bra.n_qubits() != states_ket.n_qubits()) {
        throw std::runtime_error("state_vector_bra must have same n_qubits to state_vector_ket.");
    }
    if (states_bra.batch_size() != states_ket.batch_size()) {
        throw std::runtime_error("state_vector_bra must have same batch_size to state_vector_ket.");
    }
    if (states_bra.n_qubits() != _n_qubits) {
        throw std::runtime_error(
            "PauliOperator::get_transition_amplitude: n_qubits of state_vector is too small to "
            "apply the operator");
    }
    std::uint64_t bit_flip_mask = _bit_flip_mask;
    std::uint64_t phase_flip_mask = _phase_flip_mask;
    if (bit_flip_mask == 0) {
        Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<Space>> results(
            "results", states_bra.batch_size());
        Kokkos::parallel_for(
            "get_transition_amplitude",
            Kokkos::TeamPolicy<internal::SpaceType<Space>>(
                internal::SpaceType<Space>(), states_bra.batch_size(), Kokkos::AUTO),
            KOKKOS_CLASS_LAMBDA(
                const typename Kokkos::TeamPolicy<internal::SpaceType<Space>>::member_type& team) {
                FloatType res = 0;
                std::uint64_t batch_id = team.league_rank();
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, states_bra.dim()),
                    [&](std::uint64_t state_idx, FloatType& sum) {
                        FloatType tmp =
                            (scaluq::internal::conj(states_bra._raw(batch_id, state_idx)) *
                             states_ket._raw(batch_id, state_idx))
                                .real();
                        if (Kokkos::popcount(state_idx & phase_flip_mask) & 1) tmp = -tmp;
                        sum += tmp;
                    },
                    internal::Sum<FloatType, Space>(res));
                ComplexType cres = _coef * res;
                results(batch_id) = Kokkos::complex<double>(cres.real(), cres.imag());
            });
        Kokkos::fence();
        auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
        return std::vector<StdComplex>(results_h.data(), results_h.data() + results_h.size());
    }
    std::uint64_t pivot = std::bit_width(bit_flip_mask) - 1;
    std::uint64_t global_phase_90rot_count = std::popcount(bit_flip_mask & phase_flip_mask);
    ComplexType global_phase = internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<Space>> results(
        "results", states_bra.batch_size());
    Kokkos::parallel_for(
        "get_transition_amplitude",
        Kokkos::TeamPolicy<internal::SpaceType<Space>>(
            internal::SpaceType<Space>(), states_bra.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<Space>>::member_type& team) {
            FloatType res = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, states_bra.dim() >> 1),
                [&](std::uint64_t state_idx, FloatType& sum) {
                    std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                    std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                    FloatType tmp1 = scaluq::internal::real(
                        states_bra._raw(batch_id, basis_0) *
                        scaluq::internal::conj(states_ket._raw(batch_id, basis_1)) * global_phase *
                        FloatType{2});
                    if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                    FloatType tmp2 = scaluq::internal::real(
                        states_bra._raw(batch_id, basis_1) *
                        scaluq::internal::conj(states_ket._raw(batch_id, basis_0)) * global_phase *
                        FloatType{2});
                    if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                    sum += tmp1 + tmp2;
                },
                internal::Sum<FloatType, Space>(res));
            ComplexType cres = _coef * res;
            results(batch_id) = Kokkos::complex<double>(cres.real(), cres.imag());
        });
    Kokkos::fence();
    auto results_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    return std::vector<StdComplex>(results_h.data(), results_h.data() + results_h.size());
}

template <Precision Prec, ExecutionSpace Space>
std::vector<typename PauliOperator<Prec, Space>::Triplet>
PauliOperator<Prec, Space>::get_matrix_triplets_ignoring_coef() const {
    std::uint64_t bit_mask = 0, phase_mask = 0;
    for (std::uint64_t sub_mask = _bit_flip_mask | _phase_flip_mask, idx = 0; sub_mask;
         sub_mask &= (sub_mask - 1), ++idx) {
        std::uint64_t q = std::countr_zero(sub_mask);
        if (_bit_flip_mask >> q & 1) bit_mask |= (1ULL << idx);
        if (_phase_flip_mask >> q & 1) phase_mask |= (1ULL << idx);
    }
    // count PauliID::Y
    std::uint64_t rot90_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    StdComplex rot =
        std::vector<StdComplex>{1., StdComplex(0, -1), -1., StdComplex(0, 1)}[rot90_count % 4];
    std::vector<Triplet> ret;
    std::uint64_t matrix_dim = 1ULL << std::bit_width(bit_mask | phase_mask);
    ret.reserve(matrix_dim * 2);
    for (std::uint64_t index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1 - 2 * (Kokkos::popcount(index & phase_mask) % 2);
        ret.emplace_back(index, index ^ bit_mask, rot * sign);
    }
    return ret;
}

template <Precision Prec, ExecutionSpace Space>
std::vector<typename PauliOperator<Prec, Space>::Triplet>
PauliOperator<Prec, Space>::get_full_matrix_triplets_ignoring_coef() const {
    std::uint64_t rot90_count = std::popcount(_bit_flip_mask & _phase_flip_mask);
    StdComplex rot =
        std::vector<StdComplex>{1., StdComplex(0, -1), -1., StdComplex(0, 1)}[rot90_count % 4];
    std::vector<Triplet> ret;
    std::uint64_t matrix_dim = 1ULL << _n_qubits;
    ret.reserve(matrix_dim * 2);
    for (std::uint64_t index = 0; index < matrix_dim; index++) {
        const StdComplex sign = 1 - 2 * (Kokkos::popcount(index & _phase_flip_mask) % 2);
        ret.emplace_back(index, index ^ _bit_flip_mask, rot * sign);
    }
    return ret;
}

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix PauliOperator<Prec, Space>::get_matrix() const {
    auto triplets = get_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << std::popcount(_bit_flip_mask | _phase_flip_mask);
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) =
            triplets[i].value() * static_cast<StdComplex>(_coef);
    }
    return mat;
}

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix PauliOperator<Prec, Space>::get_full_matrix() const {
    auto triplets = get_full_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << _n_qubits;
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix PauliOperator<Prec, Space>::get_matrix_ignoring_coef() const {
    auto triplets = get_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << std::popcount(_bit_flip_mask | _phase_flip_mask);
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec, ExecutionSpace Space>
ComplexMatrix PauliOperator<Prec, Space>::get_full_matrix_ignoring_coef() const {
    auto triplets = get_full_matrix_triplets_ignoring_coef();
    std::uint64_t dim = 1ULL << _n_qubits;
    ComplexMatrix mat(dim, dim);
    mat.setZero();
    for (std::size_t i = 0; i < triplets.size(); i++) {
        mat(triplets[i].row(), triplets[i].col()) = triplets[i].value();
    }
    return mat;
}

template <Precision Prec, ExecutionSpace Space>
PauliOperator<Prec, Space> PauliOperator<Prec, Space>::operator*(
    const PauliOperator& target) const {
    int extra_90rot_cnt = 0;
    auto x_left = _bit_flip_mask & ~_phase_flip_mask;
    auto y_left = _bit_flip_mask & _phase_flip_mask;
    auto z_left = _phase_flip_mask & ~_bit_flip_mask;
    auto x_right = target._bit_flip_mask & ~target._phase_flip_mask;
    auto y_right = target._bit_flip_mask & target._phase_flip_mask;
    auto z_right = target._phase_flip_mask & ~target._bit_flip_mask;
    extra_90rot_cnt += std::popcount(x_left & y_right);  // XY = iZ
    extra_90rot_cnt += std::popcount(y_left & z_right);  // YZ = iX
    extra_90rot_cnt += std::popcount(z_left & x_right);  // ZX = iY
    extra_90rot_cnt -= std::popcount(x_left & z_right);  // XZ = -iY
    extra_90rot_cnt -= std::popcount(y_left & x_right);  // YX = -iZ
    extra_90rot_cnt -= std::popcount(z_left & y_right);  // ZY = -iX
    extra_90rot_cnt %= 4;
    if (extra_90rot_cnt < 0) extra_90rot_cnt += 4;
    return PauliOperator(_n_qubits,
                         _bit_flip_mask ^ target._bit_flip_mask,
                         _phase_flip_mask ^ target._phase_flip_mask,
                         _coef * target._coef * internal::PHASE_90ROT<Prec>()[extra_90rot_cnt]);
}

template class PauliOperator<internal::Prec, internal::Space>;

}  // namespace scaluq
