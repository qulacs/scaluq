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

namespace internal {

#define DECLARE_GET_FROM_JSON(Prec, Space)                                                  \
    template <>                                                                             \
    inline std::shared_ptr<const PauliGateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();            \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();      \
        auto pauli = j.at("pauli").get<PauliOperator<Prec, Space>>();                       \
        return std::make_shared<const PauliGateImpl<Prec, Space>>(                          \
            vector_to_mask(control_qubits),                                                 \
            vector_to_mask(control_qubits, control_values),                                 \
            pauli);                                                                         \
    }                                                                                       \
    template <>                                                                             \
    inline std::shared_ptr<const PauliRotationGateImpl<Prec, Space>> get_from_json(         \
        const Json& j) {                                                                    \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();            \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();      \
        auto pauli = j.at("pauli").get<PauliOperator<Prec, Space>>();                       \
        auto angle = j.at("angle").get<double>();                                           \
        return std::make_shared<const PauliRotationGateImpl<Prec, Space>>(                  \
            vector_to_mask(control_qubits),                                                 \
            vector_to_mask(control_qubits, control_values),                                 \
            pauli,                                                                          \
            static_cast<Float<Prec>>(angle));                                               \
    }

#define INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Prec)       \
    DECLARE_GET_FROM_JSON(Prec, ExecutionSpace::Default) \
    DECLARE_GET_FROM_JSON(Prec, ExecutionSpace::Host)

#ifdef SCALUQ_BFLOAT16
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::BF16)
#endif
#ifdef SCALUQ_FLOAT16
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Precision::F64)
#endif

#undef DECLARE_GET_FROM_JSON
#undef INSTANTIATE_GET_FROM_JSON_EACH_SPACE

}  // namespace internal

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
