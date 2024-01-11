#include "./operator.hpp"

namespace qulacs {
Operator::Operator(UINT n_qubits) : _n_qubits(n_qubits) {}

std::string Operator::to_string() const {
    std::stringstream ss;
    for (auto itr = _operator_list.begin(); itr != _operator_list.end(); ++itr) {
        ss << itr->get_coef() << " " << itr->get_pauli_string();
        if (itr != prev(_operator_list.end())) {
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
    this->_operator_list.emplace_back(std::move(mpt));
}
void Operator::add_random_operator(UINT operator_count) {
    for (UINT operator_idx = 0; operator_idx < operator_count; operator_idx++) {
        std::vector<UINT> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
        for (UINT qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
            target_qubit_list[qubit_idx] = qubit_idx;
            pauli_id_list[qubit_idx] = _random.int32() & 11;
        }
        Complex coef = _random.uniform() * 2. - 1.;
        this->add_operator(PauliOperator(target_qubit_list, pauli_id_list, coef));
    }
}

Operator Operator::get_dagger() const {
    Operator quantum_operator(_n_qubits);
    for (const auto& pauli : this->_operator_list) {
        quantum_operator.add_operator(pauli.get_dagger());
    }
    return quantum_operator;
}

void Operator::apply_to_state(StateVector& state_vector) const {
    for (const auto& pauli : _operator_list) {
        pauli.apply_to_state(state_vector);
    }
}

Complex Operator::get_expectation_value(const StateVector& state_vector) const {
    if (_n_qubits > state_vector.n_qubits()) {
        throw std::runtime_error(
            "Operator::get_expectation_value: n_qubits of state_vector is too small");
    }
    Complex res = 0.;
    for (const auto& pauli : _operator_list) {
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
    for (const auto& pauli : _operator_list) {
        res += pauli.get_transition_amplitude(state_vector_bra, state_vector_ket);
    }
    return res;
}

}  // namespace qulacs
