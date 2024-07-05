#pragma once

#include <random>
#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace scaluq {
class Operator {
    struct OperatorData {
        std::vector<PauliOperator> _terms;
        UINT _n_qubits;
        bool _is_hermitian = true;
        explicit OperatorData(UINT n_qubits) : _n_qubits(n_qubits) {}
        OperatorData(UINT n_qubits, const std::vector<PauliOperator>& terms);
        void add_operator(const PauliOperator& mpt);
        void add_operator(PauliOperator&& mpt);
        void reserve(UINT size) { _terms.reserve(size); }
    };
    std::shared_ptr<const OperatorData> _ptr;

public:
    explicit Operator(UINT n_qubits) : _ptr(std::make_shared<OperatorData>(n_qubits)) {}
    explicit Operator(const OperatorData& data) : _ptr(std::make_shared<OperatorData>(data)) {}
    Operator(UINT n_qubits, const std::vector<PauliOperator>& terms)
        : _ptr(std::make_shared<OperatorData>(n_qubits, terms)) {}

    static Operator random_operator(UINT n_qubits,
                                    UINT operator_count,
                                    UINT seed = std::random_device()());

    [[nodiscard]] inline bool is_hermitian() { return _ptr->_is_hermitian; }
    [[nodiscard]] inline UINT n_qubits() { return _ptr->_n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator>& terms() const { return _ptr->_terms; }
    [[nodiscard]] std::string to_string() const;

    Operator optimize() const;

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_expectation_value_loop(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_arnoldi_method(const StateVector& state,
                                                                         UINT iter_count,
                                                                         Complex mu = 0.) const;
    // not implemented yet
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_power_method(const StateVector& state,
                                                                       UINT iter_count,
                                                                       Complex mu = 0.) const;

    Operator operator*(Complex coef) const;
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1; }
    Operator operator+(const Operator& target) const;
    Operator operator-(const Operator& target) const { return *this + target * -1; }
    Operator operator*(const Operator& target) const;
    Operator operator+(const PauliOperator& pauli) const;
    Operator operator-(const PauliOperator& pauli) const { return *this + pauli * -1; }
    Operator operator*(const PauliOperator& pauli) const;
};
}  // namespace scaluq
