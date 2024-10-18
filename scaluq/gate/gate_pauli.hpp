#pragma once

#include <vector>

#include "../operator/apply_pauli.hpp"
#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
class PauliGateImpl : public GateBase<Fp> {
    const PauliOperator<Fp> _pauli;

public:
    PauliGateImpl(std::uint64_t control_mask, const PauliOperator<Fp>& pauli)
        : GateBase<Fp>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli) {}

    PauliOperator<Fp> pauli() const { return _pauli; };
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli(
            this->_control_mask, bit_flip_mask, phase_flip_mask, _pauli.coef(), state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        auto controls = this->control_qubit_list();
        ss << indent << "Gate Type: Pauli\n";
        ss << indent << "  Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        ss << "}\n";
        ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
        return ss.str();
    }
};

template <std::floating_point Fp>
class PauliRotationGateImpl : public GateBase<Fp> {
    const PauliOperator<Fp> _pauli;
    const Fp _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask, const PauliOperator<Fp>& pauli, Fp angle)
        : GateBase<Fp>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator<Fp> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }
    Fp angle() const { return _angle; }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl<Fp>>(
            this->_control_mask, _pauli, -_angle);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override {
        internal::ComplexMatrix<Fp> mat = this->_pauli.get_matrix_ignoring_coef();
        Complex<Fp> true_angle = _angle * _pauli.coef();
        StdComplex<Fp> imag_unit(0, 1);
        mat = (StdComplex<Fp>)Kokkos::cos(-true_angle / 2) *
                  internal::ComplexMatrix<Fp>::Identity(mat.rows(), mat.cols()) +
              imag_unit * (StdComplex<Fp>)Kokkos::sin(-true_angle / 2) * mat;
        return mat;
    }
    void update_quantum_state(StateVector<Fp>& state_vector) const override {
        auto [bit_flip_mask, phase_flip_mask] = _pauli.get_XZ_mask_representation();
        apply_pauli_rotation(this->_control_mask,
                             bit_flip_mask,
                             phase_flip_mask,
                             _pauli.coef(),
                             _angle,
                             state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        auto controls = this->control_qubit_list();
        ss << indent << "Gate Type: PauliRotation\n";
        ss << indent << "  Angle: " << _angle << "\n";
        ss << indent << "  Control Qubits: {";
        for (std::uint32_t i = 0; i < controls.size(); ++i)
            ss << controls[i] << (i == controls.size() - 1 ? "" : ", ");
        ss << "}\n";
        ss << indent << "  Pauli Operator: \"" << _pauli.get_pauli_string() << "\"";
        return ss.str();
    }
};
}  // namespace internal

template <std::floating_point Fp>
using PauliGate = internal::GatePtr<internal::PauliGateImpl<Fp>>;
template <std::floating_point Fp>
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_pauli_hpp(nb::module_& m) {
    DEF_GATE(PauliGate,
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
             "gate to "
             "each of qubit.");
    DEF_GATE(PauliRotationGate,
             "Specific class of multi-qubit pauli-rotation gate, represented as "
             "$e^{-i\\frac{\\mathrm{angle}}{2}P}$.");
}
}  // namespace internal
#endif
}  // namespace scaluq
