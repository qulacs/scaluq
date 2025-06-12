#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
class PauliGateImpl : public GateBase<Prec, Space> {
    const PauliOperator<Prec, Space> _pauli;

public:
    PauliGateImpl(std::uint64_t control_mask,
                  std::uint64_t control_value_mask,
                  const PauliOperator<Prec, Space>& pauli)
        : GateBase<Prec, Space>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, control_value_mask),
          _pauli(pauli) {}

    PauliOperator<Prec, Space> pauli() const { return _pauli; };
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Pauli"},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"pauli", this->pauli()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class PauliRotationGateImpl : public GateBase<Prec, Space> {
    const PauliOperator<Prec, Space> _pauli;
    const Float<Prec> _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          const PauliOperator<Prec, Space>& pauli,
                          Float<Prec> angle)
        : GateBase<Prec, Space>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, control_value_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator<Prec, Space> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }
    double angle() const { return static_cast<double>(_angle); }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl<Prec, Space>>(
            this->_control_mask, this->_control_value_mask, _pauli, -_angle);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "PauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"pauli", this->pauli()},
                 {"angle", this->angle()}};
    }
};
}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using PauliGate = internal::GatePtr<internal::PauliGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl<Prec, Space>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_pauli_hpp(nb::module_& m, nb::class_<Gate<Prec, Space>>& gate_base_def) {
    DEF_GATE(PauliGate,
             Prec,
             Space,
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
             "gate to "
             "each of qubit.",
             gate_base_def);
    DEF_GATE(PauliRotationGate,
             Prec,
             Space,
             "Specific class of multi-qubit pauli-rotation gate, represented as "
             "$e^{-i\\frac{\\theta}{2}P}$.",
             gate_base_def);
}
}  // namespace internal
#endif
}  // namespace scaluq
