#pragma once

#include <vector>

#include "../operator/pauli_operator.hpp"
#include "../util/utility.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp, ExecutionSpace Sp>
class PauliGateImpl : public GateBase<Fp, Sp> {
    const PauliOperator<Fp, Sp> _pauli;

public:
    PauliGateImpl(std::uint64_t control_mask, const PauliOperator<Fp, Sp>& pauli)
        : GateBase<Fp, Sp>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli) {}

    PauliOperator<Fp, Sp> pauli() const { return _pauli; };
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{
            {"type", "Pauli"}, {"control", this->control_qubit_list()}, {"pauli", this->pauli()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class PauliRotationGateImpl : public GateBase<Fp, Sp> {
    const PauliOperator<Fp, Sp> _pauli;
    const Fp _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask, const PauliOperator<Fp, Sp>& pauli, Fp angle)
        : GateBase<Fp, Sp>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator<Fp, Sp> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }
    Fp angle() const { return _angle; }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl<Fp, Sp>>(
            this->_control_mask, _pauli, -_angle);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "PauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"pauli", this->pauli()},
                 {"angle", this->angle()}};
    }
};
}  // namespace internal

template <std::floating_point Fp, ExecutionSpace Sp>
using PauliGate = internal::GatePtr<internal::PauliGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl<Fp, Sp>>;

namespace internal {
/*#define DECLARE_GET_FROM_JSON_PAULIGATE_WITH_TYPE(Type)                                      \
    template <>                                                                              \
    inline std::shared_ptr<const PauliGateImpl<Type>> get_from_json(const Json& j) {         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        auto pauli = j.at("pauli").get<PauliOperator<Type>>();                               \
        return std::make_shared<const PauliGateImpl<Type>>(vector_to_mask(controls), pauli); \
    }                                                                                        \
    template <>                                                                              \
    inline std::shared_ptr<const PauliRotationGateImpl<Type>> get_from_json(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        auto pauli = j.at("pauli").get<PauliOperator<Type>>();                               \
        auto angle = j.at("angle").get<Type>();                                              \
        return std::make_shared<const PauliRotationGateImpl<Type>>(                          \
            vector_to_mask(controls), pauli, angle);                                         \
    }

DECLARE_GET_FROM_JSON_PAULIGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_PAULIGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_PAULIGATE_WITH_TYPE*/

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_gate_gate_pauli_hpp(nb::module_& m) {
    DEF_GATE(PauliGate,
             Fp,
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
             "gate to "
             "each of qubit.");
    DEF_GATE(PauliRotationGate,
             Fp,
             "Specific class of multi-qubit pauli-rotation gate, represented as "
             "$e^{-i\\frac{\\mathrm{angle}}{2}P}$.");
}
}  // namespace internal
#endif
}  // namespace scaluq
