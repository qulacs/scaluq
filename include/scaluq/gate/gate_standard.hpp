#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp>
class IGateImpl : public GateBase<Fp> {
public:
    IGateImpl() : GateBase<Fp>(0, 0) {}

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class GlobalPhaseGateImpl : public GateBase<Fp> {
protected:
    Fp _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask, Fp phase)
        : GateBase<Fp>(0, control_mask), _phase(phase){};

    [[nodiscard]] Fp phase() const { return _phase; }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Fp>>(this->_control_mask, -_phase);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class RotationGateBase : public GateBase<Fp> {
protected:
    Fp _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Fp angle)
        : GateBase<Fp>(target_mask, control_mask), _angle(angle) {}

    Fp angle() const { return _angle; }
};

template <std::floating_point Fp>
class XGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class YGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class ZGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class HGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SGateImpl;
template <std::floating_point Fp>
class SdagGateImpl;
template <std::floating_point Fp>
class TGateImpl;
template <std::floating_point Fp>
class TdagGateImpl;
template <std::floating_point Fp>
class SqrtXGateImpl;
template <std::floating_point Fp>
class SqrtXdagGateImpl;
template <std::floating_point Fp>
class SqrtYGateImpl;
template <std::floating_point Fp>
class SqrtYdagGateImpl;

template <std::floating_point Fp>
class SGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SdagGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SdagGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class TGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class TdagGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SqrtXGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Fp>>(this->_target_mask,
                                                            this->_control_mask);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SqrtXdagGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SqrtYGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Fp>>(this->_target_mask,
                                                            this->_control_mask);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SqrtYdagGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Fp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class P0GateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class P1GateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class RXGateImpl : public RotationGateBase<Fp> {
public:
    using RotationGateBase<Fp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class RYGateImpl : public RotationGateBase<Fp> {
public:
    using RotationGateBase<Fp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class RZGateImpl : public RotationGateBase<Fp> {
public:
    using RotationGateBase<Fp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class U1GateImpl : public GateBase<Fp> {
    Fp _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp lambda)
        : GateBase<Fp>(target_mask, control_mask), _lambda(lambda) {}

    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -_lambda);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};
template <std::floating_point Fp>
class U2GateImpl : public GateBase<Fp> {
    Fp _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp phi, Fp lambda)
        : GateBase<Fp>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    Fp phi() const { return _phi; }
    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Fp>>(
            this->_target_mask,
            this->_control_mask,
            static_cast<Fp>(-_lambda - Kokkos::numbers::pi),
            static_cast<Fp>(-_phi + Kokkos::numbers::pi));
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class U3GateImpl : public GateBase<Fp> {
    Fp _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp theta, Fp phi, Fp lambda)
        : GateBase<Fp>(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}

    Fp theta() const { return _theta; }
    Fp phi() const { return _phi; }
    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Fp>>(
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

template <std::floating_point Fp>
class SwapGateImpl : public GateBase<Fp> {
public:
    using GateBase<Fp>::GateBase;

    std::shared_ptr<const GateBase<Fp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp>& state_vector) const override;

    std::string to_string(const std::string& indent) const override;
};

}  // namespace internal

template <std::floating_point Fp>
using IGate = internal::GatePtr<internal::IGateImpl<Fp>>;
template <std::floating_point Fp>
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl<Fp>>;
template <std::floating_point Fp>
using XGate = internal::GatePtr<internal::XGateImpl<Fp>>;
template <std::floating_point Fp>
using YGate = internal::GatePtr<internal::YGateImpl<Fp>>;
template <std::floating_point Fp>
using ZGate = internal::GatePtr<internal::ZGateImpl<Fp>>;
template <std::floating_point Fp>
using HGate = internal::GatePtr<internal::HGateImpl<Fp>>;
template <std::floating_point Fp>
using SGate = internal::GatePtr<internal::SGateImpl<Fp>>;
template <std::floating_point Fp>
using SdagGate = internal::GatePtr<internal::SdagGateImpl<Fp>>;
template <std::floating_point Fp>
using TGate = internal::GatePtr<internal::TGateImpl<Fp>>;
template <std::floating_point Fp>
using TdagGate = internal::GatePtr<internal::TdagGateImpl<Fp>>;
template <std::floating_point Fp>
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl<Fp>>;
template <std::floating_point Fp>
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl<Fp>>;
template <std::floating_point Fp>
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl<Fp>>;
template <std::floating_point Fp>
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl<Fp>>;
template <std::floating_point Fp>
using P0Gate = internal::GatePtr<internal::P0GateImpl<Fp>>;
template <std::floating_point Fp>
using P1Gate = internal::GatePtr<internal::P1GateImpl<Fp>>;
template <std::floating_point Fp>
using RXGate = internal::GatePtr<internal::RXGateImpl<Fp>>;
template <std::floating_point Fp>
using RYGate = internal::GatePtr<internal::RYGateImpl<Fp>>;
template <std::floating_point Fp>
using RZGate = internal::GatePtr<internal::RZGateImpl<Fp>>;
template <std::floating_point Fp>
using U1Gate = internal::GatePtr<internal::U1GateImpl<Fp>>;
template <std::floating_point Fp>
using U2Gate = internal::GatePtr<internal::U2GateImpl<Fp>>;
template <std::floating_point Fp>
using U3Gate = internal::GatePtr<internal::U3GateImpl<Fp>>;
template <std::floating_point Fp>
using SwapGate = internal::GatePtr<internal::SwapGateImpl<Fp>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp>
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, Fp, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             Fp,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate<Fp>& gate) { return gate->phase(); },
            "Get `phase` property");
    DEF_GATE(XGate, Fp, "Specific class of Pauli-X gate.");
    DEF_GATE(YGate, Fp, "Specific class of Pauli-Y gate.");
    DEF_GATE(ZGate, Fp, "Specific class of Pauli-Z gate.");
    DEF_GATE(HGate, Fp, "Specific class of Hadamard gate.");
    DEF_GATE(SGate,
             Fp,
             "Specific class of S gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "i\n\\end{bmatrix}$.");
    DEF_GATE(SdagGate, Fp, "Specific class of inverse of S gate.");
    DEF_GATE(TGate,
             Fp,
             "Specific class of T gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "e^{i\\pi/4}\n\\end{bmatrix}$.");
    DEF_GATE(TdagGate, Fp, "Specific class of inverse of T gate.");
    DEF_GATE(
        SqrtXGate,
        Fp,
        "Specific class of sqrt(X) gate, represented as $\\begin{ bmatrix }\n1+i & 1-i\\\\\n1-i "
        "& 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtXdagGate, Fp, "Specific class of inverse of sqrt(X) gate.");
    DEF_GATE(SqrtYGate,
             Fp,
             "Specific class of sqrt(Y) gate, represented as $\\begin{ bmatrix }\n1+i & -1-i "
             "\\\\\n1+i & 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtYdagGate, Fp, "Specific class of inverse of sqrt(Y) gate.");
    DEF_GATE(
        P0Gate,
        Fp,
        "Specific class of projection gate to $\\ket{0}$.\n\n.. note:: This gate is not unitary.");
    DEF_GATE(
        P1Gate,
        Fp,
        "Specific class of projection gate to $\\ket{1}$.\n\n.. note:: This gate is not unitary.");

#define DEF_ROTATION_GATE(GATE_TYPE, FLOAT, DESCRIPTION)                \
    DEF_GATE(GATE_TYPE, FLOAT, DESCRIPTION)                             \
        .def(                                                           \
            "angle",                                                    \
            [](const GATE_TYPE<FLOAT>& gate) { return gate->angle(); }, \
            "Get `angle` property.")

    DEF_ROTATION_GATE(
        RXGate,
        Fp,
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}X}$.");
    DEF_ROTATION_GATE(
        RYGate,
        Fp,
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Y}$.");
    DEF_ROTATION_GATE(
        RZGate,
        Fp,
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Z}$.");

    DEF_GATE(U1Gate,
             Fp,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\begin{bmatrix}\n1 & 0\\\\\n0 & e^{i\\lambda}\n\\end{bmatrix}$.")
        .def(
            "lambda_",
            [](const U1Gate<Fp>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Fp,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate<Fp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Fp>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U3Gate,
             Fp,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, "
             "represented as "
             "$\\begin{bmatrix}\n\\cos \\frac{\\theta}{2} & "
             "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\\n"
             "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
             "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}\n\\end{bmatrix}$.")
        .def(
            "theta", [](const U3Gate<Fp>& gate) { return gate->theta(); }, "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Fp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Fp>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(SwapGate, Fp, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
