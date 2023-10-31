#pragma once

#include <vector>

#include "../types.hpp"
#include "../util/random.hpp"
#include "pauli_operator.hpp"

namespace qulacs {
class GeneralQuantumOperator {
public:
    GeneralQuantumOperator(UINT n_qubits);

    [[nodiscard]] inline bool is_hermitian() { return _is_hermitian; }
    [[nodiscard]] inline UINT n_qubits() { return _n_qubits; }
    [[nodiscard]] inline const std::vector<PauliOperator>& operator_list() const {
        return _operator_list;
    }
    [[nodiscard]] std::string to_string() const;

    void add_operator(const PauliOperator& mpt);
    void add_operator(PauliOperator&& mpt);
    void add_random_operator(const UINT operator_count = 1);

    [[nodiscard]] GeneralQuantumOperator get_dagger() const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_arnoldi_method(const StateVector& state,
                                                                         UINT iter_count,
                                                                         Complex mu = 0.) const;
    [[nodiscard]] Complex solve_gound_state_eigenvalue_by_power_method(const StateVector& state,
                                                                       UINT iter_count,
                                                                       Complex mu = 0.) const;

    [[nodiscard]] StateVector apply_to_state(const StateVector& state_vector) const;

private:
    std::vector<PauliOperator> _operator_list;
    UINT _n_qubits;
    bool _is_hermitian;
    Random _random;
};
}  // namespace qulacs
