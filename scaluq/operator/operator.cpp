#include "./operator.hpp"

#include <ranges>

#include "../constant.hpp"
#include "../util/utility.hpp"

namespace scaluq {
Operator::Operator(std::uint64_t n_qubits) : _n_qubits(n_qubits) {}

std::string Operator::to_string() const {
    std::stringstream ss;
    for (auto itr = _terms.begin(); itr != _terms.end(); ++itr) {
        ss << itr->coef() << " " << itr->get_pauli_string();
        if (itr != prev(_terms.end())) {
            ss << " + ";
        }
    }
    return ss.str();
}

void Operator::add_operator(const PauliOperator& mpt) { add_operator(PauliOperator{mpt}); }
void Operator::add_operator(PauliOperator&& mpt) {
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
void Operator::add_random_operator(std::uint64_t operator_count, std::uint64_t seed) {
    Random random(seed);
    for (std::uint64_t operator_idx = 0; operator_idx < operator_count; operator_idx++) {
        std::vector<std::uint64_t> target_qubit_list(_n_qubits), pauli_id_list(_n_qubits);
        for (std::uint64_t qubit_idx = 0; qubit_idx < _n_qubits; qubit_idx++) {
            target_qubit_list[qubit_idx] = qubit_idx;
            pauli_id_list[qubit_idx] = random.int32() & 0b11;
        }
        Complex coef = random.uniform() * 2. - 1.;
        this->add_operator(PauliOperator(target_qubit_list, pauli_id_list, coef));
    }
}

void Operator::optimize() {
    std::map<std::tuple<std::uint64_t, std::uint64_t>, Complex> pauli_and_coef;
    for (const auto& pauli : _terms) {
        pauli_and_coef[pauli.get_XZ_mask_representation()] += pauli.coef();
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
