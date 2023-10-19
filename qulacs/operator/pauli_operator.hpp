#pragma once

#include <string_view>
#include <vector>

#include "../state/state_vector.hpp"
#include "../types.hpp"

class PauliOperator {
    std::vector<UINT> _target_qubit_list, _pauli_id_list;
    Complex _coef;

public:
    PauliOperator(const std::string_view& pauli_string, Complex coef);
    PauliOperator(const std::vector<UINT>& target_qubit_list,
                  std::vector<UINT>& pauli_id_list,
                  Complex coef);
    PauliOperator(const std::vector<UINT>& pauli_id_par_qubit, Complex coef);

    [[nodiscard]] Complex get_coef() const;
    [[nodiscard]] std::string get_pauli_string() const;

    void change_coef(Complex new_coef);
    void add_single_pauli(UINT target_qubit, UINT pauli_id);

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    PauliOperator& operator*=(const PauliOperator& target);
    PauliOperator operator*(const PauliOperator& target) const;

    PauliOperator& operator*=(Complex target);
    PauliOperator operator*(Complex target) const;
};
