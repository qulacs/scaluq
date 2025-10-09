#include <scaluq/operator/operator_batched.hpp>

#include "../prec_space.hpp"
#include "apply_pauli.hpp"

namespace scaluq {

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::copy() const {
    OperatorBatched<internal::Prec, internal::Space> copy_operator;
    copy_operator._row_ptr = Kokkos::View<std::uint64_t*, Kokkos::SharedSpace>(
        Kokkos::ViewAllocateWithoutInitializing("row_ptr"), _row_ptr.extent(0));
    copy_operator._ops = Kokkos::View<Pauli*, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("ops"), _ops.extent(0));
    Kokkos::deep_copy(copy_operator._row_ptr, _row_ptr);
    Kokkos::deep_copy(copy_operator._ops, _ops);
    return copy_operator;
}

template <>
void OperatorBatched<internal::Prec, internal::Space>::load(
    const std::vector<std::vector<PauliOperator<internal::Prec, internal::Space>>>& terms) {
    std::vector<std::uint64_t> row_ptr_h;
    row_ptr_h.push_back(0);
    for (const auto& op : terms) {
        row_ptr_h.push_back(row_ptr_h.back() + op.size());
    }
    auto row_ptr_h_view = internal::wrapped_host_view(row_ptr_h);
    Kokkos::deep_copy(_row_ptr, row_ptr_h_view);
    _ops = Kokkos::View<Pauli*, ExecutionSpaceType>("operator_ops", row_ptr_h.back());
    std::vector<PauliOperator<internal::Prec, internal::Space>> ops_h;
    for (const auto& op : terms) {
        ops_h.insert(ops_h.end(), op.begin(), op.end());
    }
    auto ops_h_view = internal::wrapped_host_view(ops_h);
    Kokkos::deep_copy(_ops, ops_h_view);
}

template <>
void OperatorBatched<internal::Prec, internal::Space>::load(
    const std::vector<Operator<internal::Prec, internal::Space>>& terms) {
    std::vector<std::uint64_t> row_ptr_h;
    row_ptr_h.push_back(0);
    for (const auto& op : terms) {
        row_ptr_h.push_back(row_ptr_h.back() + op.n_terms());
    }
    auto row_ptr_h_view = internal::wrapped_host_view(row_ptr_h);
    Kokkos::deep_copy(_row_ptr, row_ptr_h_view);
    _ops = Kokkos::View<Pauli*, ExecutionSpaceType>("operator_ops", row_ptr_h.back());
    for (std::uint64_t i = 0; i < terms.size(); ++i) {
        auto term_ops = terms[i]._terms;
        Kokkos::deep_copy(Kokkos::subview(_ops, Kokkos::make_pair(row_ptr_h[i], row_ptr_h[i + 1])),
                          term_ops);
    }
}

template <>
StateVectorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::get_applied_states(
    const StateVector<internal::Prec, internal::Space>& state_vector,
    std::uint64_t batch_size) const {
    auto res = StateVectorBatched<internal::Prec, internal::Space>::uninitialized_state(
        _row_ptr.extent(0) - 1, state_vector.n_qubits());
    res.set_state_vector(state_vector);
    internal::apply_pauli<internal::Prec, internal::Space>(0, 0, _ops, _row_ptr, res, batch_size);
    return res;
}

template <>
std::vector<StdComplex> OperatorBatched<internal::Prec, internal::Space>::get_expectation_value(
    const StateVector<internal::Prec, internal::Space>& state_vector) const {
    std::uint64_t dim = state_vector.dim();
    Kokkos::View<Kokkos::complex<double>*, ExecutionSpaceType> res(
        Kokkos::ViewAllocateWithoutInitializing("expectation_value"), _row_ptr.extent(0) - 1);
    Kokkos::parallel_for(
        "get_expectation_value",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            ComplexType res_lcl = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(
                    team, _row_ptr[batch_id + 1] - _row_ptr[batch_id], dim >> 1),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    term_id += _row_ptr[batch_id];
                    auto bit_flip_mask = _ops[term_id]._bit_flip_mask;
                    auto phase_flip_mask = _ops[term_id]._phase_flip_mask;
                    ComplexType coef = _ops[term_id]._coef;
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        ComplexType tmp1 = (scaluq::internal::conj(state_vector._raw[state_idx1]) *
                                            state_vector._raw[state_idx1]);
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        ComplexType tmp2 = (scaluq::internal::conj(state_vector._raw[state_idx2]) *
                                            state_vector._raw[state_idx2]);
                        if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    } else {
                        std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                        std::uint64_t global_phase_90rot_count =
                            Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                        ComplexType global_phase =
                            internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                        std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(state_idx, pivot);
                        std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                        ComplexType tmp1 = scaluq::internal::conj(state_vector._raw[basis_1]) *
                                           state_vector._raw[basis_0] * global_phase;
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        ComplexType tmp2 = scaluq::internal::conj(state_vector._raw[basis_0]) *
                                           state_vector._raw[basis_1] * global_phase;
                        if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    }
                },
                res_lcl);
            Kokkos::single(Kokkos::PerTeam(team), [&] {
                res[batch_id] = Kokkos::complex<double>(res_lcl.real(), res_lcl.imag());
            });
        });
    Kokkos::fence();
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
std::vector<StdComplex> OperatorBatched<internal::Prec, internal::Space>::get_transition_amplitude(
    const StateVector<internal::Prec, internal::Space>& state_vector_bra,
    const StateVector<internal::Prec, internal::Space>& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
            "state_vector_ket must be same");
    }
    Kokkos::View<Kokkos::complex<double>*, ExecutionSpaceType> res(
        Kokkos::ViewAllocateWithoutInitializing("expectation_value"), _row_ptr.extent(0) - 1);
    std::uint64_t dim = state_vector_bra.dim();

    Kokkos::parallel_for(
        "get_transition_amplitude",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            ComplexType res_lcl = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(
                    team, _row_ptr[batch_id + 1] - _row_ptr[batch_id], dim >> 1),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    term_id += _row_ptr[batch_id];
                    auto bit_flip_mask = _ops[term_id]._bit_flip_mask;
                    auto phase_flip_mask = _ops[term_id]._phase_flip_mask;
                    ComplexType coef = _ops[term_id]._coef;
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        ComplexType tmp1 =
                            (scaluq::internal::conj(state_vector_bra._raw[state_idx1]) *
                             state_vector_ket._raw[state_idx1]);
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        ComplexType tmp2 =
                            (scaluq::internal::conj(state_vector_bra._raw[state_idx2]) *
                             state_vector_ket._raw[state_idx2]);
                        if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    } else {
                        std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                        std::uint64_t global_phase_90rot_count =
                            Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                        ComplexType global_phase =
                            internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                        std::uint64_t basis_0 =
                            internal::insert_zero_to_basis_index(state_idx, pivot);
                        std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                        ComplexType tmp1 = scaluq::internal::conj(state_vector_bra._raw[basis_1]) *
                                           state_vector_ket._raw[basis_0] * global_phase;
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        ComplexType tmp2 = scaluq::internal::conj(state_vector_bra._raw[basis_0]) *
                                           state_vector_ket._raw[basis_1] * global_phase;
                        if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    }
                },
                res_lcl);
            Kokkos::single(Kokkos::PerTeam(team), [&] {
                res[batch_id] = Kokkos::complex<double>(res_lcl.real(), res_lcl.imag());
            });
        });
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::get_dagger() const {
    OperatorBatched<internal::Prec, internal::Space> res = this->copy();
    Kokkos::parallel_for(
        "get_dagger",
        Kokkos::RangePolicy<ExecutionSpaceType>(0, res._ops.extent(0)),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { res._ops[i] = res._ops[i].get_dagger(); });
    return res;
}

template <>
Operator<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::get_operator_at(std::uint64_t index) const {
    if (index >= _row_ptr.extent(0) - 1) {
        throw std::out_of_range("OperatorBatched::get_operator_at: index out of range");
    }
    std::uint64_t begin = _row_ptr(index);
    std::uint64_t end = _row_ptr(index + 1);
    auto ops_subview = Kokkos::subview(_ops, std::make_pair(begin, end));
    Operator<internal::Prec, internal::Space> res(end - begin);
    Kokkos::deep_copy(res._terms, ops_subview);
    return res;
}

template <>
std::vector<Operator<internal::Prec, internal::Space>>
OperatorBatched<internal::Prec, internal::Space>::get_operators() const {
    std::vector<Operator<internal::Prec, internal::Space>> res(_row_ptr.extent(0) - 1);
    for (std::uint64_t i = 0; i < _row_ptr.extent(0) - 1; ++i) {
        res[i] = get_operator_at(i);
    }
    return res;
}

template <>
std::string OperatorBatched<internal::Prec, internal::Space>::to_string() const {
    std::string res;
    for (std::uint64_t i = 0; i < _row_ptr.extent(0) - 1; ++i) {
        auto op = get_operator_at(i);
        res += "Operator " + std::to_string(i) + ":\n";
        res += op.to_string();
    }
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::operator*(
    const std::vector<StdComplex>& coef) const {
    if (coef.size() != _row_ptr.extent(0) - 1) {
        throw std::runtime_error(
            "OperatorBatched::operator*: size of coef must be equal to batch size");
    }
    std::vector<ComplexType> tmp(coef.begin(), coef.end());
    auto coef_view = internal::convert_vector_to_view<ComplexType, internal::Space>(tmp);
    OperatorBatched<internal::Prec, internal::Space> res = this->copy();
    Kokkos::parallel_for(
        "operator*",
        Kokkos::RangePolicy<ExecutionSpaceType>(0, res._ops.extent(0)),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { res._ops[i] *= coef_view[i]; });
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>&
OperatorBatched<internal::Prec, internal::Space>::operator*=(const std::vector<StdComplex>& coef) {
    if (coef.size() != _row_ptr.extent(0) - 1) {
        throw std::runtime_error(
            "OperatorBatched::operator*: size of coef must be equal to batch size");
    }
    std::vector<ComplexType> tmp(coef.begin(), coef.end());
    auto coef_view = internal::convert_vector_to_view<ComplexType, internal::Space>(tmp);
    Kokkos::parallel_for(
        "operator*=",
        Kokkos::RangePolicy<ExecutionSpaceType>(0, _ops.extent(0)),
        KOKKOS_CLASS_LAMBDA(std::uint64_t i) { _ops[i] *= coef_view[i]; });
    return *this;
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::operator+(const OperatorBatched& target) const {
    if (_row_ptr.extent(0) != target._row_ptr.extent(0)) {
        throw std::runtime_error(
            "OperatorBatched::operator+: batch size of both operators must be same");
    }
    std::vector<std::uint64_t> row_ptr_h;
    row_ptr_h.push_back(0);
    for (std::uint64_t i = 0; i < _row_ptr.extent(0) - 1; ++i) {
        row_ptr_h.push_back(row_ptr_h.back() + (_row_ptr[i + 1] - _row_ptr[i]) +
                            (target._row_ptr[i + 1] - target._row_ptr[i]));
    }
    OperatorBatched<internal::Prec, internal::Space> res;
    res._row_ptr = Kokkos::View<std::uint64_t*, Kokkos::SharedSpace>(
        Kokkos::ViewAllocateWithoutInitializing("row_ptr"), row_ptr_h.size());
    Kokkos::deep_copy(res._row_ptr, internal::wrapped_host_view(row_ptr_h));
    res._ops = Kokkos::View<Pauli*, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("operator_ops"), row_ptr_h.back());
    Kokkos::parallel_for(
        "operator+",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            std::uint64_t begin1 = _row_ptr(batch_id);
            std::uint64_t end1 = _row_ptr(batch_id + 1);
            std::uint64_t begin2 = target._row_ptr(batch_id);
            std::uint64_t end2 = target._row_ptr(batch_id + 1);
            std::uint64_t begin_res = res._row_ptr(batch_id);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, end1 - begin1),
                [&](std::uint64_t i) { res._ops[begin_res + i] = _ops[begin1 + i]; });
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, end2 - begin2), [&](std::uint64_t i) {
                    res._ops[begin_res + (end1 - begin1) + i] = target._ops[begin2 + i];
                });
        });
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::operator*(const OperatorBatched& target) const {
    if (_row_ptr.extent(0) != target._row_ptr.extent(0)) {
        throw std::runtime_error(
            "OperatorBatched::operator*: batch size of both operators must be same");
    }
    std::vector<std::uint64_t> row_ptr_h;
    row_ptr_h.push_back(0);
    for (std::uint64_t i = 0; i < _row_ptr.extent(0) - 1; ++i) {
        row_ptr_h.push_back(row_ptr_h.back() + (_row_ptr[i + 1] - _row_ptr[i]) *
                                                   (target._row_ptr[i + 1] - target._row_ptr[i]));
    }
    OperatorBatched<internal::Prec, internal::Space> res;
    res._row_ptr = Kokkos::View<std::uint64_t*, Kokkos::SharedSpace>(
        Kokkos::ViewAllocateWithoutInitializing("row_ptr"), row_ptr_h.size());
    Kokkos::deep_copy(res._row_ptr, internal::wrapped_host_view(row_ptr_h));
    res._ops = Kokkos::View<Pauli*, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("operator_ops"), row_ptr_h.back());
    Kokkos::parallel_for(
        "operator+",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            std::uint64_t begin1 = _row_ptr(batch_id);
            std::uint64_t end1 = _row_ptr(batch_id + 1);
            std::uint64_t begin2 = target._row_ptr(batch_id);
            std::uint64_t end2 = target._row_ptr(batch_id + 1);
            std::uint64_t begin_res = res._row_ptr(batch_id);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, (end1 - begin1) * (end2 - begin2)),
                                 [&](std::uint64_t i) {
                                     std::uint64_t i1 = i / (end2 - begin2);
                                     std::uint64_t i2 = i % (end2 - begin2);
                                     res._ops[begin_res + i] =
                                         _ops[begin1 + i1] * target._ops[begin2 + i2];
                                 });
        });
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::operator+(
    const std::vector<PauliOperator<internal::Prec, internal::Space>>& pauli) const {
    if (_row_ptr.extent(0) != pauli.size()) {
        throw std::runtime_error(
            "OperatorBatched::operator+: batch size of both operators must be same");
    }
    std::vector<std::uint64_t> row_ptr_h;
    row_ptr_h.push_back(0);
    for (std::uint64_t i = 0; i < _row_ptr.extent(0) - 1; ++i) {
        row_ptr_h.push_back(row_ptr_h.back() + (_row_ptr[i + 1] - _row_ptr[i]) + 1);
    }
    OperatorBatched<internal::Prec, internal::Space> res;
    res._row_ptr = Kokkos::View<std::uint64_t*, Kokkos::SharedSpace>(
        Kokkos::ViewAllocateWithoutInitializing("row_ptr"), row_ptr_h.size());
    Kokkos::deep_copy(res._row_ptr, internal::wrapped_host_view(row_ptr_h));
    res._ops = Kokkos::View<Pauli*, ExecutionSpaceType>(
        Kokkos::ViewAllocateWithoutInitializing("operator_ops"), row_ptr_h.back());
    Kokkos::parallel_for(
        "operator+",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            std::uint64_t begin = _row_ptr(batch_id);
            std::uint64_t end = _row_ptr(batch_id + 1);
            std::uint64_t begin_res = res._row_ptr(batch_id);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, end - begin + 1), [&](std::uint64_t i) {
                    res._ops[begin_res + i] = (i < end - begin ? _ops[begin + i] : pauli[batch_id]);
                });
        });
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>
OperatorBatched<internal::Prec, internal::Space>::operator*(
    const std::vector<PauliOperator<internal::Prec, internal::Space>>& pauli) const {
    if (_row_ptr.extent(0) != pauli.size()) {
        throw std::runtime_error(
            "OperatorBatched::operator+: batch size of both operators must be same");
    }
    OperatorBatched<internal::Prec, internal::Space> res = this->copy();
    Kokkos::parallel_for(
        "operator+",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            std::uint64_t begin = _row_ptr(batch_id);
            std::uint64_t end = _row_ptr(batch_id + 1);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, end - begin + 1),
                [&](std::uint64_t i) { res._ops[begin + i] = _ops[begin + i] * pauli[batch_id]; });
        });
    return res;
}

template <>
OperatorBatched<internal::Prec, internal::Space>&
OperatorBatched<internal::Prec, internal::Space>::operator*=(
    const std::vector<PauliOperator<internal::Prec, internal::Space>>& pauli) {
    if (_row_ptr.extent(0) != pauli.size()) {
        throw std::runtime_error(
            "OperatorBatched::operator*: batch size of both operators must be same");
    }
    Kokkos::parallel_for(
        "operator+",
        Kokkos::TeamPolicy<ExecutionSpaceType>(_row_ptr.extent(0) - 1, Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<ExecutionSpaceType>::member_type& team) {
            std::uint64_t batch_id = team.league_rank();
            std::uint64_t begin = _row_ptr(batch_id);
            std::uint64_t end = _row_ptr(batch_id + 1);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, end - begin + 1),
                [&](std::uint64_t i) { _ops[begin + i] = _ops[begin + i] * pauli[batch_id]; });
        });
    return *this;
}

}  // namespace scaluq
