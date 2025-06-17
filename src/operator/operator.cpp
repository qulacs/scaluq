#include <scaluq/operator/operator.hpp>

#include "../prec_space.hpp"
#include "../util/math.hpp"

namespace scaluq {

template <Precision Prec, ExecutionSpace Space>
static std::tuple<Kokkos::View<std::uint64_t*, internal::SpaceType<Space>>,
                  Kokkos::View<std::uint64_t*, internal::SpaceType<Space>>,
                  Kokkos::View<internal::Complex<Prec>*, internal::SpaceType<Space>>>
prepare_mask_and_coef(const std::vector<PauliOperator<Prec, Space>>& paulis) {
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> bit_flip_mask_h("bit_flip_mask", paulis.size()),
        phase_flip_mask_h("phase_flip_mask", paulis.size());
    Kokkos::View<internal::Complex<Prec>*, Kokkos::HostSpace> coef_h("coef", paulis.size());

    Kokkos::parallel_for("prepare_mask_and_coef",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, paulis.size()),
                         [&](const std::uint64_t i) {
                             std::tie(bit_flip_mask_h(i), phase_flip_mask_h(i)) =
                                 paulis[i].get_XZ_mask_representation();
                             coef_h(i) = paulis[i].coef();
                         });
    Kokkos::fence();

    auto bit_flip_mask =
        Kokkos::create_mirror_view_and_copy(internal::SpaceType<Space>(), bit_flip_mask_h);
    auto phase_flip_mask =
        Kokkos::create_mirror_view_and_copy(internal::SpaceType<Space>(), phase_flip_mask_h);
    auto coef = Kokkos::create_mirror_view_and_copy(internal::SpaceType<Space>(), coef_h);
    return std::make_tuple(bit_flip_mask, phase_flip_mask, coef);
}

template <>
std::string Operator<internal::Prec, internal::Space>::to_string() const {
    std::stringstream ss;
    for (auto itr = _terms.begin(); itr != _terms.end(); ++itr) {
        ss << itr->coef() << " " << itr->get_pauli_string();
        if (itr != prev(_terms.end())) {
            ss << " + ";
        }
    }
    return ss.str();
}

template <>
void Operator<internal::Prec, internal::Space>::add_operator(
    PauliOperator<internal::Prec, internal::Space>&& mpt) {
    _is_hermitian &= mpt.coef().imag() == 0.;
    if (![&] {
            const auto& target_list = mpt.target_qubit_list();
            if (target_list.empty()) return true;
            return *std::max_element(target_list.begin(), target_list.end()) < _n_qubits;
        }()) {
        throw std::runtime_error(
            "Operator::add_operator: target index of pauli_operator is larger than "
            "n_qubits");
    }
    this->_terms.emplace_back(std::move(mpt));
}

template <>
void Operator<internal::Prec, internal::Space>::add_random_operator(
    const std::uint64_t operator_count, std::uint64_t seed) {
    Random random(seed);
    for (std::uint64_t operator_idx = 0; operator_idx < operator_count; operator_idx++) {
        std::vector<std::uint64_t> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
        for (std::uint64_t qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
            target_qubit_list[qubit_idx] = qubit_idx;
            pauli_id_list[qubit_idx] = random.int32() & 0b11;
        }
        StdComplex coef = random.uniform() * 2. - 1.;
        this->add_operator(
            PauliOperator<internal::Prec, internal::Space>(target_qubit_list, pauli_id_list, coef));
    }
}

template <>
void Operator<internal::Prec, internal::Space>::optimize() {
    std::map<std::tuple<std::uint64_t, std::uint64_t>, ComplexType> pauli_and_coef;
    for (const auto& pauli : _terms) {
        pauli_and_coef[pauli.get_XZ_mask_representation()] += pauli._ptr->_coef;
    }
    _terms.clear();
    for (const auto& [mask, coef] : pauli_and_coef) {
        const auto& [x_mask, z_mask] = mask;
        _terms.emplace_back(x_mask, z_mask, StdComplex(coef));
    }
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::get_dagger()
    const {
    Operator<internal::Prec, internal::Space> quantum_operator(_n_qubits);
    for (const auto& pauli : _terms) {
        quantum_operator.add_operator(pauli.get_dagger());
    }
    return quantum_operator;
}

template <>
ComplexMatrix Operator<internal::Prec, internal::Space>::get_matrix() const {
    std::uint64_t dim = 1ULL << _n_qubits;
    using Pauli = PauliOperator<internal::Prec, internal::Space>;
    std::vector<typename Pauli::Triplet> triplets;
    triplets.reserve(dim * _terms.size());
    for (const auto& term : _terms) {
        std::vector<typename Pauli::PauliID> pauli_id_par_qubit(_n_qubits, Pauli::PauliID::I);
        for (std::uint64_t i = 0; i < term._ptr->_pauli_id_list.size(); i++) {
            pauli_id_par_qubit[term._ptr->_target_qubit_list[i]] =
                static_cast<typename Pauli::PauliID>(term._ptr->_pauli_id_list[i]);
        }
        typename Pauli::Data aligned_data;
        for (std::uint64_t i = 0; i < _n_qubits; i++) {
            aligned_data.add_single_pauli(i, pauli_id_par_qubit[i]);
        }
        auto basic_triplets = Pauli(aligned_data).get_matrix_triplets_ignoring_coef();
        std::ranges::transform(
            basic_triplets, std::back_inserter(triplets), [&](const Pauli::Triplet& triplet) {
                return typename Pauli::Triplet(
                    triplet.row(),
                    triplet.col(),
                    triplet.value() * static_cast<StdComplex>(term._ptr->_coef));
            });
    }
    SparseComplexMatrix sparse(dim, dim);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    return ComplexMatrix(sparse);
}

template <>
void Operator<internal::Prec, internal::Space>::apply_to_state(
    StateVector<internal::Prec, internal::Space>& state_vector) const {
    auto res =
        StateVector<internal::Prec, internal::Space>::uninitialized_state(state_vector.n_qubits());
    res.set_zero_norm_state();
    for (const auto& term : _terms) {
        StateVector<internal::Prec, internal::Space> tmp = state_vector.copy();
        term.apply_to_state(tmp);
        res.add_state_vector_with_coef(1., tmp);
    }
    state_vector = res;
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::get_expectation_value(
    const StateVector<internal::Prec, internal::Space>& state_vector) const {
    if (_n_qubits > state_vector.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    std::uint64_t nterms = _terms.size();
    auto [bmasks, pmasks, coefs] = prepare_mask_and_coef(_terms);
    std::uint64_t dim = state_vector.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        "get_expectation_value",
        Kokkos::MDRangePolicy<internal::SpaceType<internal::Space>, Kokkos::Rank<2>>(
            {0, 0}, {nterms, dim >> 1}),
        KOKKOS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, ComplexType & res_lcl) {
            std::uint64_t bit_flip_mask = bmasks[term_id];
            std::uint64_t phase_flip_mask = pmasks[term_id];
            ComplexType coef = coefs[term_id];
            if (bit_flip_mask == 0) {
                std::uint64_t state_idx1 = state_idx << 1;
                FloatType tmp1 = (scaluq::internal::conj(state_vector._raw[state_idx1]) *
                                  state_vector._raw[state_idx1])
                                     .real();
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                std::uint64_t state_idx2 = state_idx1 | 1;
                FloatType tmp2 = (scaluq::internal::conj(state_vector._raw[state_idx2]) *
                                  state_vector._raw[state_idx2])
                                     .real();
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                std::uint64_t basis_1 = basis_0 ^ bit_flip_mask;
                FloatType tmp =
                    scaluq::internal::real(state_vector._raw[basis_0] *
                                           scaluq::internal::conj(state_vector._raw[basis_1]) *
                                           global_phase * FloatType(2));
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                res_lcl += coef * tmp;
            }
        },
        res);
    Kokkos::fence();
    return static_cast<StdComplex>(res);
}

template <>
std::vector<StdComplex> Operator<internal::Prec, internal::Space>::get_expectation_value(
    const StateVectorBatched<internal::Prec, internal::Space>& states) const {
    if (_n_qubits > states.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    std::uint64_t nterms = _terms.size();
    auto [bmasks, pmasks, coefs] = prepare_mask_and_coef(_terms);
    std::uint64_t dim = states.dim();
    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<internal::Space>> res(
        "expectation_value_res", states.batch_size());
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>(
            internal::SpaceType<internal::Space>(), states.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>::member_type&
                team) {
            ComplexType sum = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(team, nterms, dim >> 1),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    std::uint64_t bit_flip_mask = bmasks[term_id];
                    std::uint64_t phase_flip_mask = pmasks[term_id];
                    ComplexType coef = coefs[term_id];
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        FloatType tmp1 =
                            (scaluq::internal::conj(states._raw(batch_id, state_idx1)) *
                             states._raw(batch_id, state_idx1))
                                .real();
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        FloatType tmp2 =
                            (scaluq::internal::conj(states._raw(batch_id, state_idx2)) *
                             states._raw(batch_id, state_idx2))
                                .real();
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
                        FloatType tmp = scaluq::internal::real(
                            states._raw(batch_id, basis_0) *
                            scaluq::internal::conj(states._raw(batch_id, basis_1)) * global_phase *
                            FloatType(2));
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                        res_lcl += coef * tmp;
                    }
                },
                sum);
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                res(batch_id) = Kokkos::complex<double>(sum.real(), sum.imag());
            });
        });
    Kokkos::fence();
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), res);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
StdComplex Operator<internal::Prec, internal::Space>::get_transition_amplitude(
    const StateVector<internal::Prec, internal::Space>& state_vector_bra,
    const StateVector<internal::Prec, internal::Space>& state_vector_ket) const {
    if (state_vector_bra.n_qubits() != state_vector_ket.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
            "state_vector_ket must be same");
    }
    if (_n_qubits > state_vector_bra.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector is too "
            "small");
    }
    std::uint64_t nterms = _terms.size();
    auto [bmasks, pmasks, coefs] = prepare_mask_and_coef(_terms);
    std::uint64_t dim = state_vector_bra.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        "get_transition_amplitude",
        Kokkos::MDRangePolicy<internal::SpaceType<internal::Space>, Kokkos::Rank<2>>(
            {0, 0}, {nterms, dim >> 1}),
        KOKKOS_LAMBDA(std::uint64_t term_id, std::uint64_t state_idx, ComplexType & res_lcl) {
            std::uint64_t bit_flip_mask = bmasks[term_id];
            std::uint64_t phase_flip_mask = pmasks[term_id];
            ComplexType coef = coefs[term_id];
            if (bit_flip_mask == 0) {
                std::uint64_t state_idx1 = state_idx << 1;
                ComplexType tmp1 = (scaluq::internal::conj(state_vector_bra._raw[state_idx1]) *
                                    state_vector_ket._raw[state_idx1]);
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                std::uint64_t state_idx2 = state_idx1 | 1;
                ComplexType tmp2 = (scaluq::internal::conj(state_vector_bra._raw[state_idx2]) *
                                    state_vector_ket._raw[state_idx2]);
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                std::uint64_t pivot = Kokkos::bit_width(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<internal::Prec>()[global_phase_90rot_count % 4];
                std::uint64_t basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
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
        res);
    Kokkos::fence();
    return static_cast<StdComplex>(res);
}

template <>
std::vector<StdComplex> Operator<internal::Prec, internal::Space>::get_transition_amplitude(
    const StateVectorBatched<internal::Prec, internal::Space>& states_bra,
    const StateVectorBatched<internal::Prec, internal::Space>& states_ket) const {
    if (states_bra.n_qubits() != states_ket.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector_bra and "
            "state_vector_ket must be same");
    }
    if (states_bra.batch_size() != states_ket.batch_size()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: batch_size of state_vector_bra and "
            "state_vector_ket must be same");
    }
    if (_n_qubits > states_bra.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_transition_amplitude: n_qubits of state_vector is too "
            "small");
    }
    std::uint64_t nterms = _terms.size();
    auto [bmasks, pmasks, coefs] = prepare_mask_and_coef(_terms);
    std::uint64_t dim = states_bra.dim();
    Kokkos::View<Kokkos::complex<double>*, internal::SpaceType<internal::Space>> results(
        "transition_amplitude_res", states_bra.batch_size());
    Kokkos::parallel_for(
        "get_transition_amplitude",
        Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>(
            internal::SpaceType<internal::Space>(), states_bra.batch_size(), Kokkos::AUTO),
        KOKKOS_CLASS_LAMBDA(
            const typename Kokkos::TeamPolicy<internal::SpaceType<internal::Space>>::member_type&
                team) {
            ComplexType res = 0;
            std::uint64_t batch_id = team.league_rank();
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadMDRange(team, nterms, (dim >> 1)),
                [&](std::uint64_t term_id, std::uint64_t state_idx, ComplexType& res_lcl) {
                    std::uint64_t bit_flip_mask = bmasks[term_id];
                    std::uint64_t phase_flip_mask = pmasks[term_id];
                    ComplexType coef = coefs[term_id];
                    if (bit_flip_mask == 0) {
                        std::uint64_t state_idx1 = state_idx << 1;
                        ComplexType tmp1 =
                            (scaluq::internal::conj(states_bra._raw(batch_id, state_idx1)) *
                             states_ket._raw(batch_id, state_idx1));
                        if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        std::uint64_t state_idx2 = state_idx1 | 1;
                        ComplexType tmp2 =
                            (scaluq::internal::conj(states_bra._raw(batch_id, state_idx2)) *
                             states_ket._raw(batch_id, state_idx2));
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
                        ComplexType tmp1 =
                            scaluq::internal::conj(states_bra._raw(batch_id, basis_1)) *
                            states_ket._raw(batch_id, basis_0) * global_phase;
                        if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                        ComplexType tmp2 =
                            scaluq::internal::conj(states_bra._raw(batch_id, basis_0)) *
                            states_ket._raw(batch_id, basis_1) * global_phase;
                        if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                        res_lcl += coef * (tmp1 + tmp2);
                    }
                },
                res);
            Kokkos::single(Kokkos::PerTeam(team), [res, batch_id, results]() {
                results(batch_id) = Kokkos::complex<double>(res.real(), res.imag());
            });
        });
    Kokkos::fence();
    auto res_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), results);
    return std::vector<StdComplex>(res_h.data(), res_h.data() + res_h.size());
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator*=(
    StdComplex coef) {
    for (auto& pauli : _terms) {
        pauli = pauli * ComplexType(coef);
    }
    return *this;
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator+=(
    const Operator<internal::Prec, internal::Space>& target) {
    if (_n_qubits != target._n_qubits) {
        throw std::runtime_error("Operator::operator+=: n_qubits must be equal");
    }
    for (const auto& pauli : target._terms) {
        add_operator(pauli);
    }
    return *this;
}

template <>
Operator<internal::Prec, internal::Space> Operator<internal::Prec, internal::Space>::operator*(
    const Operator<internal::Prec, internal::Space>& target) const {
    if (_n_qubits != target._n_qubits) {
        throw std::runtime_error("Operator::operator+=: n_qubits must be equal");
    }
    Operator ret(_n_qubits);
    for (const auto& pauli1 : _terms) {
        for (const auto& pauli2 : target._terms) {
            ret.add_operator(pauli1 * pauli2);
        }
    }
    return ret;
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator+=(
    const PauliOperator<internal::Prec, internal::Space>& pauli) {
    add_operator(pauli);
    return *this;
}

template <>
Operator<internal::Prec, internal::Space>& Operator<internal::Prec, internal::Space>::operator*=(
    const PauliOperator<internal::Prec, internal::Space>& pauli) {
    for (auto& pauli1 : _terms) {
        pauli1 = pauli1 * pauli;
    }
    return *this;
}

template class Operator<internal::Prec, internal::Space>;

}  // namespace scaluq
