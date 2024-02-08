#pragma once

#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace qulacs {
class Operator {
public:
    explicit Operator(UINT n_qubits);

    [[nodiscard]] inline bool is_hermitian() { return _is_hermitian; }
    [[nodiscard]] inline UINT n_qubits() { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator>& terms() const { return _terms; }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator& mpt);
    void add_operator(PauliOperator&& mpt);
    void add_random_operator(const UINT operator_count = 1);
    void add_random_operator(const UINT operator_count, UINT seed);

    void optimize();

    [[nodiscard]] Operator get_dagger() const;

    // not implemented yet
    void get_matrix() const;

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
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

    Operator& operator*=(Complex coef);
    Operator operator*(Complex coef) const { return Operator(*this) *= coef; }
    inline Operator operator+() const { return *this; }
    Operator operator-() const { return *this * -1; }
    Operator& operator+=(const Operator& target);
    Operator operator+(const Operator& target) const { return Operator(*this) += target; }
    Operator& operator-=(const Operator& target) { return *this += -target; }
    Operator operator-(const Operator& target) const { return Operator(*this) -= target; }
    Operator operator*(const Operator& target) const;
    Operator& operator*=(const Operator& target) { return *this = *this * target; }
    Operator& operator+=(const PauliOperator& pauli);
    Operator operator+(const PauliOperator& pauli) const { return Operator(*this) += pauli; }
    Operator& operator-=(const PauliOperator& pauli) { return *this += pauli * -1; }
    Operator operator-(const PauliOperator& pauli) const { return Operator(*this) -= pauli; }
    Operator& operator*=(const PauliOperator& pauli);
    Operator operator*(const PauliOperator& pauli) { return Operator(*this) *= pauli; }

private:
    std::vector<PauliOperator> _terms;
    UINT _n_qubits;
    bool _is_hermitian;
    Random _random;
};
}  // namespace qulacs
