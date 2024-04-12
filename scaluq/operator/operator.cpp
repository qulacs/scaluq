#include "./operator.hpp"

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
    std::map<std::tuple<BitVector, BitVector>, Complex> pauli_and_coef;
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
    for (const auto& pauli : _terms) {
        pauli.apply_to_state(state_vector);
    }
}

Complex Operator::get_expectation_value(const StateVector& state_vector) const {
    if (_n_qubits > state_vector.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    Complex res = 0.;
    for (const auto& pauli : _terms) {
        res += pauli.get_expectation_value(state_vector);
    }
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
    Complex res;
    for (const auto& pauli : _terms) {
        res += pauli.get_transition_amplitude(state_vector_bra, state_vector_ket);
    }
    return res;
}

Operator& Operator::operator*=(Complex coef) {
    for (auto& pauli : _terms) pauli *= coef;
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
    for (auto& pauli1 : _terms) pauli1 *= pauli;
    return *this;
}

}  // namespace scaluq
