#include "./operator.hpp"

#include <ranges>

#include "../constant.hpp"
#include "../util/utility.hpp"

namespace scaluq {
Operator::Operator(UINT n_qubits) : _n_qubits(n_qubits) {}

std::string Operator::to_string() const {
    std::stringstream ss;
    for (auto itr = _terms.begin(); itr != _terms.end(); ++itr) {
        ss << itr->get_coef() << " " << itr->get_pauli_string();
        if (itr != prev(_terms.end())) {
            ss << " + ";
        }
    }
    return ss.str();
}

void Operator::add_operator(const PauliOperator& mpt) { add_operator(PauliOperator{mpt}); }
void Operator::add_operator(PauliOperator&& mpt) {
    _is_hermitian &= mpt.get_coef().imag() == 0.;
    if (![&] {
            const auto& target_list = mpt.get_target_qubit_list();
            if (target_list.empty()) return true;
            return *std::max_element(target_list.begin(), target_list.end()) < _n_qubits;
        }()) {
        throw std::runtime_error(
            "Operator::add_operator: target index of pauli_operator is larger than "
            "n_qubits");
    }
    this->_terms.emplace_back(std::move(mpt));
}
void Operator::add_random_operator(UINT operator_count, UINT seed) {
    Random random(seed);
    for (UINT operator_idx = 0; operator_idx < operator_count; operator_idx++) {
        std::vector<UINT> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
        for (UINT qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
            target_qubit_list[qubit_idx] = qubit_idx;
            pauli_id_list[qubit_idx] = random.int32() & 0b11;
        }
        Complex coef = random.uniform() * 2. - 1.;
        this->add_operator(PauliOperator(target_qubit_list, pauli_id_list, coef));
    }
}

void Operator::optimize() {
    std::map<std::tuple<internal::BitVector, internal::BitVector>, Complex> pauli_and_coef;
    for (const auto& pauli : _terms) {
        pauli_and_coef[pauli.get_XZ_mask_representation()] += pauli.get_coef();
    }
    _terms.clear();
    for (const auto& [mask, coef] : pauli_and_coef) {
        const auto& [x_mask, z_mask] = mask;
        _terms.emplace_back(x_mask, z_mask, coef);
    }
}

Operator Operator::get_dagger() const {
    Operator quantum_operator(_n_qubits);
    for (const auto& pauli : _terms) {
        quantum_operator.add_operator(pauli.get_dagger());
    }
    return quantum_operator;
}

void Operator::apply_to_state(StateVector& state_vector) const {
    StateVector res(state_vector.n_qubits());
    res.set_zero_norm_state();
    for (const auto& term : _terms) {
        StateVector tmp = state_vector.copy();
        term.apply_to_state(tmp);
        res.add_state_vector(tmp);
    }
    state_vector = res;
}

Complex Operator::get_expectation_value(const StateVector& state_vector) const {
    if (_n_qubits > state_vector.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    UINT nterms = _terms.size();
    Kokkos::View<const PauliOperator*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        terms_view(_terms.data(), nterms);
    Kokkos::View<UINT*, Kokkos::HostSpace> bmasks_host("bmasks_host", nterms);
    Kokkos::View<UINT*, Kokkos::HostSpace> pmasks_host("pmasks_host", nterms);
    Kokkos::View<Complex*, Kokkos::HostSpace> coefs_host("coefs_host", nterms);
    Kokkos::Experimental::transform(
        Kokkos::DefaultHostExecutionSpace(),
        terms_view,
        bmasks_host,
        [](const PauliOperator& pauli) { return pauli._ptr->_bit_flip_mask.data_raw()[0]; });
    Kokkos::Experimental::transform(
        Kokkos::DefaultHostExecutionSpace(),
        terms_view,
        pmasks_host,
        [](const PauliOperator& pauli) { return pauli._ptr->_phase_flip_mask.data_raw()[0]; });
    Kokkos::Experimental::transform(Kokkos::DefaultHostExecutionSpace(),
                                    terms_view,
                                    coefs_host,
                                    [](const PauliOperator& pauli) { return pauli._ptr->_coef; });
    Kokkos::View<UINT*> bmasks("bmasks", nterms);
    Kokkos::View<UINT*> pmasks("pmasks", nterms);
    Kokkos::View<Complex*> coefs("coefs", nterms);
    Kokkos::deep_copy(bmasks, bmasks_host);
    Kokkos::deep_copy(pmasks, pmasks_host);
    Kokkos::deep_copy(coefs, coefs_host);
    UINT dim = state_vector.dim();
    Complex res;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nterms, dim >> 1}),
        KOKKOS_LAMBDA(UINT term_id, UINT state_idx, Complex & res_lcl) {
            UINT bit_flip_mask = bmasks[term_id];
            UINT phase_flip_mask = pmasks[term_id];
            Complex coef = coefs[term_id];
            if (bit_flip_mask == 0) {
                UINT state_idx1 = state_idx << 1;
                double tmp1 =
                    (Kokkos::conj(state_vector._raw[state_idx1]) * state_vector._raw[state_idx1])
                        .real();
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                UINT state_idx2 = state_idx1 | 1;
                double tmp2 =
                    (Kokkos::conj(state_vector._raw[state_idx2]) * state_vector._raw[state_idx2])
                        .real();
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                UINT pivot = sizeof(UINT) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                UINT global_phase_90rot_count = Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                Complex global_phase = PHASE_90ROT().val[global_phase_90rot_count % 4];
                UINT basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                UINT basis_1 = basis_0 ^ bit_flip_mask;
                double tmp =
                    Kokkos::real(state_vector._raw[basis_0] *
                                 Kokkos::conj(state_vector._raw[basis_1]) * global_phase * 2.);
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp = -tmp;
                res_lcl += coef * tmp;
            }
        },
        res);
    Kokkos::fence();
    return res;
}

Complex Operator::get_transition_amplitude(const StateVector& state_vector_bra,
                                           const StateVector& state_vector_ket) const {
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
    UINT nterms = _terms.size();
    std::vector<UINT> bmasks_vector(nterms);
    std::vector<UINT> pmasks_vector(nterms);
    std::vector<Complex> coefs_vector(nterms);
    std::transform(
        _terms.begin(), _terms.end(), bmasks_vector.begin(), [](const PauliOperator& pauli) {
            return pauli._ptr->_bit_flip_mask.data_raw()[0];
        });
    std::transform(
        _terms.begin(), _terms.end(), pmasks_vector.begin(), [](const PauliOperator& pauli) {
            return pauli._ptr->_phase_flip_mask.data_raw()[0];
        });
    std::transform(
        _terms.begin(), _terms.end(), coefs_vector.begin(), [](const PauliOperator& pauli) {
            return pauli._ptr->_coef;
        });
    Kokkos::View<UINT*> bmasks = internal::convert_host_vector_to_device_view(bmasks_vector);
    Kokkos::View<UINT*> pmasks = internal::convert_host_vector_to_device_view(pmasks_vector);
    Kokkos::View<Complex*> coefs = internal::convert_host_vector_to_device_view(coefs_vector);
    UINT dim = state_vector_bra.dim();
    Complex res;
    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nterms, dim >> 1}),
        KOKKOS_LAMBDA(UINT term_id, UINT state_idx, Complex & res_lcl) {
            UINT bit_flip_mask = bmasks[term_id];
            UINT phase_flip_mask = pmasks[term_id];
            Complex coef = coefs[term_id];
            if (bit_flip_mask == 0) {
                UINT state_idx1 = state_idx << 1;
                Complex tmp1 = (Kokkos::conj(state_vector_bra._raw[state_idx1]) *
                                state_vector_ket._raw[state_idx1]);
                if (Kokkos::popcount(state_idx1 & phase_flip_mask) & 1) tmp1 = -tmp1;
                UINT state_idx2 = state_idx1 | 1;
                Complex tmp2 = (Kokkos::conj(state_vector_bra._raw[state_idx2]) *
                                state_vector_ket._raw[state_idx2]);
                if (Kokkos::popcount(state_idx2 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            } else {
                UINT pivot = sizeof(UINT) * 8 - Kokkos::countl_zero(bit_flip_mask) - 1;
                UINT global_phase_90rot_count = Kokkos::popcount(bit_flip_mask & phase_flip_mask);
                Complex global_phase = PHASE_90ROT().val[global_phase_90rot_count % 4];
                UINT basis_0 = internal::insert_zero_to_basis_index(state_idx, pivot);
                UINT basis_1 = basis_0 ^ bit_flip_mask;
                Complex tmp1 = Kokkos::conj(state_vector_bra._raw[basis_1]) *
                               state_vector_ket._raw[basis_0] * global_phase;
                if (Kokkos::popcount(basis_0 & phase_flip_mask) & 1) tmp1 = -tmp1;
                Complex tmp2 = Kokkos::conj(state_vector_bra._raw[basis_0]) *
                               state_vector_ket._raw[basis_1] * global_phase;
                if (Kokkos::popcount(basis_1 & phase_flip_mask) & 1) tmp2 = -tmp2;
                res_lcl += coef * (tmp1 + tmp2);
            }
        },
        res);
    Kokkos::fence();
    return res;
}

Operator& Operator::operator*=(Complex coef) {
    for (auto& pauli : _terms) {
        pauli = pauli * coef;
    }
    return *this;
}
Operator& Operator::operator+=(const Operator& target) {
    if (_n_qubits != target._n_qubits) {
        throw std::runtime_error("Operator::oeprator+=: n_qubits must be equal");
    }
    for (const auto& pauli : target._terms) {
        add_operator(pauli);
    }
    return *this;
}
Operator Operator::operator*(const Operator& target) const {
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
Operator& Operator::operator+=(const PauliOperator& pauli) {
    add_operator(pauli);
    return *this;
}
Operator& Operator::operator*=(const PauliOperator& pauli) {
    for (auto& pauli1 : _terms) {
        pauli1 = pauli1 * pauli;
    }
    return *this;
}

}  // namespace scaluq
