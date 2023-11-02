#pragma once

#include <string_view>
#include <vector>

#include "../state/state_vector.hpp"
#include "../types.hpp"

namespace qulacs {
class PauliOperator {
    std::vector<UINT> _target_qubit_list, _pauli_id_list;
    Complex _coef;
    UINT _bit_flip_mask, _phase_flip_mask;

public:
    PauliOperator(Complex coef = 1.);
    PauliOperator(std::string_view pauli_string, Complex coef = 1.);
    PauliOperator(const std::vector<UINT>& target_qubit_list,
                  const std::vector<UINT>& pauli_id_list,
                  Complex coef = 1.);
    PauliOperator(const std::vector<UINT>& pauli_id_par_qubit, Complex coef = 1.);
    PauliOperator(UINT bit_flip_mask, UINT phase_flip_mask, Complex coef = 1.);

    [[nodiscard]] inline Complex get_coef() const { return _coef; }
    [[nodiscard]] inline const std::vector<UINT>& get_target_qubit_list() const {
        return _target_qubit_list;
    }
    [[nodiscard]] inline const std::vector<UINT>& get_pauli_id_list() const {
        return _pauli_id_list;
    }
    [[nodiscard]] inline std::tuple<UINT, UINT> get_XZ_mask_representation() const {
        return {_bit_flip_mask, _phase_flip_mask};
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] inline PauliOperator get_dagger() const {
        return PauliOperator(_target_qubit_list, _pauli_id_list, std::conj(_coef));
    }

    inline void change_coef(Complex new_coef) { _coef = new_coef; }
    void add_single_pauli(UINT target_qubit, UINT pauli_id);

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const;
    PauliOperator& operator*=(const PauliOperator& target);

    inline PauliOperator& operator*=(Complex target) {
        _coef *= target;
        return *this;
    };
    [[nodiscard]] inline PauliOperator operator*(Complex target) const {
        return PauliOperator(*this) * target;
    }
};

}  // namespace qulacs
