#include "./general_quantum_operator.hpp"

namespace qulacs {
GeneralQuantumOperator::GeneralQuantumOperator(UINT n_qubits) : _n_qubits(n_qubits) {}

std::string GeneralQuantumOperator::to_string() const {
    std::stringstream ss;
    for (auto itr = _operator_list.begin(); itr != _operator_list.end(); ++itr) {
        ss << itr->get_coef() << " " << itr->get_pauli_string();
        if (itr != prev(_operator_list.end())) {
            ss << " + ";
        }
    }
    return ss.str();
}

void GeneralQuantumOperator::add_operator(const PauliOperator& mpt) {
    add_operator(PauliOperator{mpt});
}
void GeneralQuantumOperator::add_operator(PauliOperator&& mpt) {
    _is_hermitian &= mpt.get_coef().imag() == 0.;
    if (![&] {
            const auto& target_list = mpt.get_target_qubit_list();
            if (target_list.empty()) return false;
            return *std::max_element(target_list.begin(), target_list.end()) < _n_qubits;
        }()) {
        throw std::runtime_error(
            "GeneralQuantumOperator::add_operator: target index of pauli_operator is larger than "
            "n_qubits");
    }
    this->_operator_list.emplace_back(std::move(mpt));
}
void GeneralQuantumOperator::add_random_operator(UINT operator_count) {
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

GeneralQuantumOperator GeneralQuantumOperator::get_dagger() const {
    GeneralQuantumOperator quantum_operator(_n_qubits);
    for (auto pauli : this->_operator_list) {
        quantum_operator.add_operator(pauli.get_dagger());
    }
    return quantum_operator;
}

Complex GeneralQuantumOperator::get_expectation_value(const StateVector& state_vector) const {
    Complex res;
    Kokkos::parallel_reduce(
        "expectation_value",
        _operator_list.size(),
        KOKKOS_CLASS_LAMBDA(const UINT& operator_idx, Complex& sum) {
            sum += _operator_list[operator_idx].get_expectation_value(state_vector);
        },
        res);
    return res;
}
Complex GeneralQuantumOperator::get_transition_amplitude(
    const StateVector& state_vector_bra, const StateVector& state_vector_ket) const {
    Complex res;
    Kokkos::parallel_reduce(
        "expectation_value",
        _operator_list.size(),
        KOKKOS_CLASS_LAMBDA(const UINT& operator_idx, Complex& sum) {
            sum += _operator_list[operator_idx].get_transition_amplitude(state_vector_bra,
                                                                         state_vector_ket);
        },
        res);
    return res;
}

}  // namespace qulacs
