#include <scaluq/operator/operator.hpp>

#include "../util/template.hpp"

namespace scaluq {
template <Precision Prec, ExecutionSpace Space>
std::string Operator<Prec, Space>::to_string() const {
    std::stringstream ss;
    for (auto itr = _terms.begin(); itr != _terms.end(); ++itr) {
        ss << itr->coef() << " " << itr->get_pauli_string();
        if (itr != prev(_terms.end())) {
            ss << " + ";
        }
    }
    return ss.str();
}

template <Precision Prec, ExecutionSpace Space>
void Operator<Prec, Space>::add_operator(PauliOperator<Prec, Space>&& mpt) {
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

template <Precision Prec, ExecutionSpace Space>
void Operator<Prec, Space>::add_random_operator(const std::uint64_t operator_count,
                                                std::uint64_t seed) {
    Random random(seed);
    for (std::uint64_t operator_idx = 0; operator_idx < operator_count; operator_idx++) {
        std::vector<std::uint64_t> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
        for (std::uint64_t qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
            target_qubit_list[qubit_idx] = qubit_idx;
            pauli_id_list[qubit_idx] = random.int32() & 0b11;
        }
        StdComplex coef = random.uniform() * 2. - 1.;
        this->add_operator(PauliOperator<Prec, Space>(target_qubit_list, pauli_id_list, coef));
    }
}

template <Precision Prec, ExecutionSpace Space>
void Operator<Prec, Space>::optimize() {
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

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space> Operator<Prec, Space>::get_dagger() const {
    Operator<Prec, Space> quantum_operator(_n_qubits);
    for (const auto& pauli : _terms) {
        quantum_operator.add_operator(pauli.get_dagger());
    }
    return quantum_operator;
}

template <Precision Prec, ExecutionSpace Space>
internal::ComplexMatrix Operator<Prec, Space>::get_matrix() {
    std::uint64_t dim = 1ULL << _n_qubits;
    using Pauli = PauliOperator<Prec, Space>;
    std::vector<typename Pauli::Triplet> triplets;
    triplets.reserve(dim, _terms.size());
    for (const auto& term : _terms) {
        std::vector<Pauli::PauliID> pauli_id_par_qubit(_n_qubits, Pauli::PauliID::I);
        for (std::uint64_t i = 0; i < term._ptr->_pauli_id_list; i++) {
            pauli_id_par_qubit[term._ptr->_target_qubit_list[i]] = term._ptr->_pauli_id_list[i];
        }
        Pauli::Data aligned_data;
        for (std::uint64_t i = 0; i < _n_qubits; i++) {
            aligned_data.add_single_pauli(i, pauli_id_par_qubit[i]);
        }
        auto basic_triplets = Pauli(aligned_data).get_matrix_triplets_ignoring_coef();
        for (auto& triplet : basic_triplets) {
            triplet.value() *= static_cast<StdComplex>(term._ptr->coef);
        }
        std::ranges::copy(basic_triplets, std::back_inserter(triplets));
    }
    internal::SparseMatrix sparse(dim, dim);
    sparse.setFromTriplets(triplets);
    return internal::ComplexMatrix(sparse);
}

template <Precision Prec, ExecutionSpace Space>
void Operator<Prec, Space>::apply_to_state(StateVector<Prec, Space>& state_vector) const {
    StateVector<Prec, Space> res(state_vector.n_qubits());
    res.set_zero_norm_state();
    for (const auto& term : _terms) {
        StateVector<Prec, Space> tmp = state_vector.copy();
        term.apply_to_state(tmp);
        res.add_state_vector_with_coef(1., tmp);
    }
    state_vector = res;
}

template <Precision Prec, ExecutionSpace Space>
StdComplex Operator<Prec, Space>::get_expectation_value(
    const StateVector<Prec, Space>& state_vector) const {
    if (_n_qubits > state_vector.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    std::uint64_t nterms = _terms.size();
    Kokkos::View<const PauliOperator<Prec, Space>*,
                 Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        terms_view(_terms.data(), nterms);
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> bmasks_host("bmasks_host", nterms);
    Kokkos::View<std::uint64_t*, Kokkos::HostSpace> pmasks_host("pmasks_host", nterms);
    Kokkos::View<ComplexType*, Kokkos::HostSpace> coefs_host("coefs_host", nterms);
    Kokkos::Experimental::transform(
        Kokkos::DefaultHostExecutionSpace(),
        terms_view,
        bmasks_host,
        [](const PauliOperator<Prec, Space>& pauli) { return pauli._ptr->_bit_flip_mask; });
    Kokkos::Experimental::transform(
        Kokkos::DefaultHostExecutionSpace(),
        terms_view,
        pmasks_host,
        [](const PauliOperator<Prec, Space>& pauli) { return pauli._ptr->_phase_flip_mask; });
    Kokkos::Experimental::transform(
        Kokkos::DefaultHostExecutionSpace(),
        terms_view,
        coefs_host,
        [](const PauliOperator<Prec, Space>& pauli) { return pauli._ptr->_coef; });
    Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> bmasks("bmasks", nterms);
    Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> pmasks("pmasks", nterms);
    Kokkos::View<ComplexType*, internal::SpaceType<Space>> coefs("coefs", nterms);
    Kokkos::deep_copy(bmasks, bmasks_host);
    Kokkos::deep_copy(pmasks, pmasks_host);
    Kokkos::deep_copy(coefs, coefs_host);
    std::uint64_t dim = state_vector.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {nterms, dim >> 1}),
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
                std::uint64_t pivot =
                    sizeof(std::uint64_t) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
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

template <Precision Prec, ExecutionSpace Space>
StdComplex Operator<Prec, Space>::get_transition_amplitude(
    const StateVector<Prec, Space>& state_vector_bra,
    const StateVector<Prec, Space>& state_vector_ket) const {
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
    std::vector<std::uint64_t> bmasks_vector(nterms);
    std::vector<std::uint64_t> pmasks_vector(nterms);
    std::vector<ComplexType> coefs_vector(nterms);
    std::ranges::transform(
        _terms, bmasks_vector.begin(), [](const PauliOperator<Prec, Space>& pauli) {
            return pauli._ptr->_bit_flip_mask;
        });
    std::ranges::transform(
        _terms, pmasks_vector.begin(), [](const PauliOperator<Prec, Space>& pauli) {
            return pauli._ptr->_phase_flip_mask;
        });
    std::ranges::transform(
        _terms, coefs_vector.begin(), [](const PauliOperator<Prec, Space>& pauli) {
            return pauli._ptr->_coef;
        });
    Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> bmasks =
        internal::convert_vector_to_view<std::uint64_t, Space>(bmasks_vector);
    Kokkos::View<std::uint64_t*, internal::SpaceType<Space>> pmasks =
        internal::convert_vector_to_view<std::uint64_t, Space>(pmasks_vector);
    Kokkos::View<ComplexType*, internal::SpaceType<Space>> coefs =
        internal::convert_vector_to_view<ComplexType, Space>(coefs_vector);
    std::uint64_t dim = state_vector_bra.dim();
    ComplexType res;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<internal::SpaceType<Space>, Kokkos::Rank<2>>({0, 0},
                                                                           {nterms, dim >> 1}),
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
                std::uint64_t pivot =
                    sizeof(std::uint64_t) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                std::uint64_t global_phase_90rot_count =
                    Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                ComplexType global_phase =
                    internal::PHASE_90ROT<Prec>()[global_phase_90rot_count % 4];
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

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space>& Operator<Prec, Space>::operator*=(StdComplex coef) {
    for (auto& pauli : _terms) {
        pauli = pauli * ComplexType(coef);
    }
    return *this;
}

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space>& Operator<Prec, Space>::operator+=(const Operator<Prec, Space>& target) {
    if (_n_qubits != target._n_qubits) {
        throw std::runtime_error("Operator::oeprator+=: n_qubits must be equal");
    }
    for (const auto& pauli : target._terms) {
        add_operator(pauli);
    }
    return *this;
}

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space> Operator<Prec, Space>::operator*(const Operator<Prec, Space>& target) const {
    if (_n_qubits != target._n_qubits) {
        throw std::runtime_error("Operator::oeprator+=: n_qubits must be equal");
    }
    Operator ret(_n_qubits);
    for (const auto& pauli1 : _terms) {
        for (const auto& pauli2 : target._terms) {
            ret.add_operator(pauli1 * pauli2);
        }
    }
    return ret;
}

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space>& Operator<Prec, Space>::operator+=(const PauliOperator<Prec, Space>& pauli) {
    add_operator(pauli);
    return *this;
}

template <Precision Prec, ExecutionSpace Space>
Operator<Prec, Space>& Operator<Prec, Space>::operator*=(const PauliOperator<Prec, Space>& pauli) {
    for (auto& pauli1 : _terms) {
        pauli1 = pauli1 * pauli;
    }
    return *this;
}

SCALUQ_DECLARE_CLASS_FOR_PRECISION_AND_EXECUTION_SPACE(Operator)

}  // namespace scaluq
