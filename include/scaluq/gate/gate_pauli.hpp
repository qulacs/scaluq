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
    PauliGateImpl(std::uint64_t control_mask, const PauliOperator<Prec>& pauli)
        : GateBase<Prec>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli) {}

    PauliOperator<Prec> pauli() const { return _pauli; };
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override { return this->_pauli.get_matrix(); }

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{
            {"type", "Pauli"}, {"control", this->control_qubit_list()}, {"pauli", this->pauli()}};
    }
};

template <Precision Prec>
class PauliRotationGateImpl : public GateBase<Prec> {
    const PauliOperator<Prec> _pauli;
    const Float<Prec> _angle;

public:
    PauliRotationGateImpl(std::uint64_t control_mask,
                          const PauliOperator<Prec>& pauli,
                          Float<Prec> angle)
        : GateBase<Prec>(vector_to_mask<false>(pauli.target_qubit_list()), control_mask),
          _pauli(pauli),
          _angle(angle) {}

    PauliOperator<Prec> pauli() const { return _pauli; }
    std::vector<std::uint64_t> pauli_id_list() const { return _pauli.pauli_id_list(); }
    double angle() const { return _angle; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const PauliRotationGateImpl<Prec>>(
            this->_control_mask, _pauli, -_angle);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "PauliRotation"},
                 {"control", this->control_qubit_list()},
                 {"pauli", this->pauli()},
                 {"angle", this->angle()}};
    }
};
}  // namespace internal

template <Precision Prec>
using PauliGate = internal::GatePtr<internal::PauliGateImpl<Prec>>;
template <Precision Prec>
using PauliRotationGate = internal::GatePtr<internal::PauliRotationGateImpl<Prec>>;

namespace internal {
#define DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION(Prec)                                 \
    template <>                                                                              \
    inline std::shared_ptr<const PauliGateImpl<Prec>> get_from_json(const Json& j) {         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        auto pauli = j.at("pauli").get<PauliOperator<Prec>>();                               \
        return std::make_shared<const PauliGateImpl<Prec>>(vector_to_mask(controls), pauli); \
    }                                                                                        \
    template <>                                                                              \
    inline std::shared_ptr<const PauliRotationGateImpl<Prec>> get_from_json(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        auto pauli = j.at("pauli").get<PauliOperator<Prec>>();                               \
        auto angle = j.at("angle").get<double>();                                            \
        return std::make_shared<const PauliRotationGateImpl<Prec>>(                          \
            vector_to_mask(controls), pauli, static_cast<Float<Prec>>(angle));               \
    }

#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_PAULIGATE_WITH_PRECISION

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_pauli_hpp(nb::module_& m) {
    DEF_GATE(PauliGate,
             Prec,
             "Specific class of multi-qubit pauli gate, which applies single-qubit Pauli "
             "gate to "
             "each of qubit.");
    DEF_GATE(PauliRotationGate,
             Prec,
             "Specific class of multi-qubit pauli-rotation gate, represented as "
             "$e^{-i\\frac{\\mathrm{angle}}{2}P}$.");
}
}  // namespace internal
#endif
}  // namespace scaluq
