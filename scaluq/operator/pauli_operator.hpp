#pragma once

#include <string_view>
#include <vector>

#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "../util/bit_vector.hpp"

namespace scaluq {
class PauliOperator {
    friend class Operator;

public:
    class Data {
        friend class PauliOperator;
        friend class Operator;
        std::vector<std::uint64_t> _target_qubit_list, _pauli_id_list;
        Complex _coef;
        internal::BitVector _bit_flip_mask, _phase_flip_mask;

    public:
        explicit Data(Complex coef = 1.);
        Data(std::string_view pauli_string, Complex coef = 1.);
        Data(const std::vector<std::uint64_t>& target_qubit_list,
             const std::vector<std::uint64_t>& pauli_id_list,
             Complex coef = 1.);
        Data(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex coef = 1.);
        Data(const std::vector<bool>& bit_flip_mask,
             const std::vector<bool>& phase_flip_mask,
             Complex coef);
        void add_single_pauli(std::uint64_t target_qubit, std::uint64_t pauli_id);
        Complex coef() const { return _coef; }
        void set_coef(Complex c) { _coef = c; }
        const std::vector<std::uint64_t>& target_qubit_list() const { return _target_qubit_list; }
        const std::vector<std::uint64_t>& pauli_id_list() const { return _pauli_id_list; }
        std::tuple<std::vector<bool>, std::vector<bool>> get_XZ_mask_representation() const {
            return {_bit_flip_mask, _phase_flip_mask};
        }
    };

private:
    std::shared_ptr<const Data> _ptr;

public:
    enum PauliID : std::uint64_t { I, X, Y, Z };

    explicit PauliOperator(Complex coef = 1.) : _ptr(std::make_shared<const Data>(coef)) {}
    explicit PauliOperator(Data data) : _ptr(std::make_shared<const Data>(data)) {}
    PauliOperator(std::string_view pauli_string, Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_string, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& target_qubit_list,
                  const std::vector<std::uint64_t>& pauli_id_list,
                  Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(target_qubit_list, pauli_id_list, coef)) {}
    PauliOperator(const std::vector<std::uint64_t>& pauli_id_par_qubit, Complex coef = 1.)
        : _ptr(std::make_shared<const Data>(pauli_id_par_qubit, coef)) {}
    PauliOperator(const std::vector<bool>& bit_flip_mask,
                  const std::vector<bool>& phase_flip_mask,
                  Complex coef)
        : _ptr(std::make_shared<const Data>(bit_flip_mask, phase_flip_mask, coef)) {}

    [[nodiscard]] inline Complex coef() const { return _ptr->coef(); }
    [[nodiscard]] inline const std::vector<std::uint64_t>& target_qubit_list() const {
        return _ptr->target_qubit_list();
    }
    [[nodiscard]] inline const std::vector<std::uint64_t>& pauli_id_list() const {
        return _ptr->pauli_id_list();
    }
    [[nodiscard]] inline std::tuple<std::vector<bool>, std::vector<bool>>
    get_XZ_mask_representation() const {
        return _ptr->get_XZ_mask_representation();
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] inline PauliOperator get_dagger() const {
        return PauliOperator(
            _ptr->_target_qubit_list, _ptr->_pauli_id_list, Kokkos::conj(_ptr->_coef));
    }
    [[nodiscard]] std::uint64_t get_qubit_count() const {
        if (_ptr->_target_qubit_list.empty()) return 0;
        return std::ranges::max(_ptr->_target_qubit_list) + 1;
    }

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;
    [[nodiscard]] ComplexMatrix get_matrix() const;
    [[nodiscard]] ComplexMatrix get_matrix_ignoring_coef() const;

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const;
    [[nodiscard]] inline PauliOperator operator*(Complex target) const {
        return PauliOperator(_ptr->_target_qubit_list, _ptr->_pauli_id_list, _ptr->_coef * target);
    }
};

}  // namespace scaluq
