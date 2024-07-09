#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "../state/state_vector.hpp"
#include "../types.hpp"
#include "../util/bit_vector.hpp"
#include "../util/random.hpp"

namespace scaluq {

class PauliOperator {
    friend class Operator;

public:
    struct Data {
        static constexpr UINT I = 0, X = 1, Y = 2, Z = 3;
        std::vector<UINT> _target_qubit_list, _pauli_id_list;
        Complex _coef;
        internal::BitVector _bit_flip_mask, _phase_flip_mask;
        explicit Data(Complex coef = 1.);
        Data(std::string_view pauli_string, Complex coef = 1.);
        Data(const std::vector<UINT>& target_qubit_list,
             const std::vector<UINT>& pauli_id_list,
             Complex coef = 1.);
        Data(const std::vector<UINT>& pauli_id_par_qubit, Complex coef = 1.);
        Data(const std::vector<bool>& bit_flip_mask,
             const std::vector<bool>& phase_flip_mask,
             Complex coef);
        void add_single_pauli(UINT target_qubit, UINT pauli_id);
        void reserve(UINT size) {
            _target_qubit_list.reserve(size);
            _pauli_id_list.reserve(size);
        }
    };

private:
    std::shared_ptr<const Data> _ptr;

public:
    explicit PauliOperator(Complex coef = 1.) : _ptr(std::make_shared<Data>(coef)) {}
    explicit PauliOperator(const Data& data) : _ptr(std::make_shared<Data>(data)) {}
    PauliOperator(std::string_view pauli_string, Complex coef = 1.)
        : _ptr(std::make_shared<Data>(pauli_string, coef)) {}
    PauliOperator(const std::vector<UINT>& target_qubit_list,
                  const std::vector<UINT>& pauli_id_list,
                  Complex coef = 1.)
        : _ptr(std::make_shared<Data>(target_qubit_list, pauli_id_list, coef)) {}
    PauliOperator(const std::vector<UINT>& pauli_id_par_qubit, Complex coef = 1.)
        : _ptr(std::make_shared<Data>(pauli_id_par_qubit, coef)) {}
    PauliOperator(const std::vector<bool>& bit_flip_mask,
                  const std::vector<bool>& phase_flip_mask,
                  Complex coef)
        : _ptr(std::make_shared<Data>(bit_flip_mask, phase_flip_mask, coef)) {}

    [[nodiscard]] inline Complex get_coef() const { return _ptr->_coef; }
    [[nodiscard]] inline const std::vector<UINT>& get_target_qubit_list() const {
        return _ptr->_target_qubit_list;
    }
    [[nodiscard]] inline const std::vector<UINT>& get_pauli_id_list() const {
        return _ptr->_pauli_id_list;
    }
    [[nodiscard]] inline std::tuple<std::vector<bool>, std::vector<bool>>
    get_XZ_mask_representation() const {
        return {_ptr->_bit_flip_mask, _ptr->_phase_flip_mask};
    }
    [[nodiscard]] std::string get_pauli_string() const;
    [[nodiscard]] inline PauliOperator get_dagger() const {
        return PauliOperator(
            _ptr->_target_qubit_list, _ptr->_pauli_id_list, Kokkos::conj(_ptr->_coef));
    }
    [[nodiscard]] UINT get_qubit_count() const {
        if (_ptr->_target_qubit_list.empty()) return 0;
        return std::ranges::max(_ptr->_target_qubit_list) + 1;
    }

    void apply_to_state(StateVector& state_vector) const;

    [[nodiscard]] Complex get_expectation_value(const StateVector& state_vector) const;
    [[nodiscard]] Complex get_transition_amplitude(const StateVector& state_vector_bra,
                                                   const StateVector& state_vector_ket) const;

    [[nodiscard]] PauliOperator operator*(const PauliOperator& target) const;
    [[nodiscard]] inline PauliOperator operator*(Complex target) const {
        auto cp(*_ptr);
        cp._coef *= target;
        return PauliOperator(cp);
    }
};

}  // namespace scaluq
