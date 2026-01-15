#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec>
class PauliGateImpl : public GateBase<Prec> {
    const PauliOperator<Prec> _pauli;

public:
    PauliGateImpl(std::uint64_t control_mask,
                  std::uint64_t control_value_mask,
                  const PauliOperator<Prec>& pauli)
        : GateBase<Prec>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, control_value_mask),
          _pauli(pauli) {}

    PauliOperator<Prec> pauli() const { return _pauli; };
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& states) const override;
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::HostSerialSpace>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::HostSerialSpace>& states) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& states) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Pauli"},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"pauli", this->pauli()}};
    }
};

template <Precision Prec>
class PauliRotationGateImpl : public GateBase<Prec> {
    const PauliOperator<Prec> _pauli;
    const Float<Prec> _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask,
                          std::uint64_t control_value_mask,
                          const PauliOperator<Prec>& pauli,
                          Float<Prec> angle)
        : GateBase<Prec>(
              vector_to_mask<false>(pauli.target_qubit_list()), control_mask, control_value_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator<Prec> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }
    double angle() const { return static_cast<double>(_angle); }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl<Prec>>(
            this->_control_mask, this->_control_value_mask, _pauli, -_angle);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& states) const override;
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::HostSerialSpace>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::HostSerialSpace>& states) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& states) const override;
#endif  // SCALUQ_USE_CUDA

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

template <Precision Prec>
using PauliGate = internal::GatePtr<internal::PauliGateImpl<Prec>>;
template <Precision Prec>
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_pauli_hpp(nb::module_& m, nb::class_<Gate<Prec>>& gate_base_def) {
    bind_specific_gate<PauliGate<Prec>, Prec>(
        m,
        gate_base_def,
        "PauliGate",
        "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
        "gate to "
        "each of qubit.");
    bind_specific_gate<PauliRotationGate<Prec>, Prec>(
        m,
        gate_base_def,
        "PauliRotationGate",
        "Specific class of multi-qubit pauli-rotation gate, represented as "
        "$e^{-i\\frac{\\theta}{2}P}$.");
}
}  // namespace internal
#endif
}  // namespace scaluq
