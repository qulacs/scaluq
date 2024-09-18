#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

class IGateImpl : public GateBase {
public:
    IGateImpl() : GateBase(0, 0) {}

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        return internal::ComplexMatrix::Identity(1, 1);
    }

    void update_quantum_state(StateVector& state_vector) const override {
        i_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: I\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class GlobalPhaseGateImpl : public GateBase {
protected:
    double _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask, double phase)
        : GateBase(0, control_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    Gate get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl>(_control_mask, -_phase);
    }
    internal::ComplexMatrix get_matrix() const override {
        return internal::ComplexMatrix::Identity(1, 1) * std::exp(std::complex<double>(0, _phase));
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        global_phase_gate(_target_mask, _control_mask, _phase, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: GlobalPhase\n";
        ss << indent << "  Phase: " << _phase << "\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class RotationGateBase : public GateBase {
protected:
    double _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, double angle)
        : GateBase(target_mask, control_mask), _angle(angle) {}

    double angle() const { return _angle; }
};

class XGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, 1, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        x_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: X\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class YGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, -1i, 1i, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        y_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Y\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class ZGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        z_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Z\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class HGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 1, 1, -1;
        mat /= std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        h_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: H\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class SGateImpl;
class SdagGateImpl;
class TGateImpl;
class TdagGateImpl;
class SqrtXGateImpl;
class SqrtXdagGateImpl;
class SqrtYGateImpl;
class SqrtYdagGateImpl;

class SGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override;
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        s_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: S\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class SdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SGateImpl>(_target_mask, _control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sdag_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Sdag\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};
// for resolving dependency issues
inline Gate SGateImpl::get_inverse() const {
    return std::make_shared<const SdagGateImpl>(_target_mask, _control_mask);
}

class TGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override;
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. + 1i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        t_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: T\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class TdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const TGateImpl>(_target_mask, _control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. - 1.i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        tdag_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Tdag\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};
// for resolving dependency issues
inline Gate TGateImpl::get_inverse() const {
    return std::make_shared<const TdagGateImpl>(_target_mask, _control_mask);
}

class SqrtXGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override;
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtx_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtX\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class SqrtXdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl>(_target_mask, _control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtxdag_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtXdag\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};
// for resolving dependency issues
inline Gate SqrtXGateImpl::get_inverse() const {
    return std::make_shared<const SqrtXdagGateImpl>(_target_mask, _control_mask);
}

class SqrtYGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override;
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrty_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtY\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class SqrtYdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl>(_target_mask, _control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtydag_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtYdag\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};
// for resolving dependency issues
inline Gate SqrtYGateImpl::get_inverse() const {
    return std::make_shared<const SqrtYdagGateImpl>(_target_mask, _control_mask);
}

class P0GateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        p0_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: P0\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class P1GateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        p1_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: P1\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class RXGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RXGateImpl>(_target_mask, _control_mask, -_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -1i * std::sin(_angle / 2), -1i * std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        rx_gate(_target_mask, _control_mask, _angle, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RX\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class RYGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RYGateImpl>(_target_mask, _control_mask, -_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -std::sin(_angle / 2), std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        ry_gate(_target_mask, _control_mask, _angle, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RY\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class RZGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RZGateImpl>(_target_mask, _control_mask, -_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * _angle), 0, 0, std::exp(0.5i * _angle);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        rz_gate(_target_mask, _control_mask, _angle, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RZ\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class U1GateImpl : public GateBase {
    double _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, double lambda)
        : GateBase(target_mask, control_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    Gate get_inverse() const override {
        return std::make_shared<const U1GateImpl>(_target_mask, _control_mask, -_lambda);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, std::exp(1i * _lambda);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        u1_gate(_target_mask, _control_mask, _lambda, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U1\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};
class U2GateImpl : public GateBase {
    double _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, double phi, double lambda)
        : GateBase(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate get_inverse() const override {
        return std::make_shared<const U2GateImpl>(_target_mask,
                                                  _control_mask,
                                                  -_lambda - Kokkos::numbers::pi,
                                                  -_phi + Kokkos::numbers::pi);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(Kokkos::numbers::pi / 4.),
            -std::exp(1i * _lambda) * std::sin(Kokkos::numbers::pi / 4.),
            std::exp(1i * _phi) * std::sin(Kokkos::numbers::pi / 4.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(Kokkos::numbers::pi / 4.);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        u2_gate(_target_mask, _control_mask, _phi, _lambda, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U2\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class U3GateImpl : public GateBase {
    double _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               double theta,
               double phi,
               double lambda)
        : GateBase(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate get_inverse() const override {
        return std::make_shared<const U3GateImpl>(
            _target_mask, _control_mask, -_theta, -_lambda, -_phi);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(_theta / 2.), -std::exp(1i * _lambda) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(_theta / 2.);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        u3_gate(_target_mask, _control_mask, _theta, _phi, _lambda, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U3\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

class SwapGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat = internal::ComplexMatrix::Identity(1 << 2, 1 << 2);
        mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        swap_gate(_target_mask, _control_mask, state_vector);
    }

    void update_quantum_state(StateVectorBatched& states) const override {}

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Swap\n";
        ss << get_qubit_info_as_string(indent);
        return ss.str();
    }
};

}  // namespace internal

using IGate = internal::GatePtr<internal::IGateImpl>;
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl>;

using XGate = internal::GatePtr<internal::XGateImpl>;
using YGate = internal::GatePtr<internal::YGateImpl>;
using ZGate = internal::GatePtr<internal::ZGateImpl>;
using HGate = internal::GatePtr<internal::HGateImpl>;
using SGate = internal::GatePtr<internal::SGateImpl>;
using SdagGate = internal::GatePtr<internal::SdagGateImpl>;
using TGate = internal::GatePtr<internal::TGateImpl>;
using TdagGate = internal::GatePtr<internal::TdagGateImpl>;
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl>;
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl>;
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl>;
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl>;
using P0Gate = internal::GatePtr<internal::P0GateImpl>;
using P1Gate = internal::GatePtr<internal::P1GateImpl>;
using RXGate = internal::GatePtr<internal::RXGateImpl>;
using RYGate = internal::GatePtr<internal::RYGateImpl>;
using RZGate = internal::GatePtr<internal::RZGateImpl>;
using U1Gate = internal::GatePtr<internal::U1GateImpl>;
using U2Gate = internal::GatePtr<internal::U2GateImpl>;
using U3Gate = internal::GatePtr<internal::U3GateImpl>;

using SwapGate = internal::GatePtr<internal::SwapGateImpl>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate& gate) { return gate->phase(); },
            "Get `phase` property");
    DEF_GATE(XGate, "Specific class of Pauli-X gate.");
    DEF_GATE(YGate, "Specific class of Pauli-Y gate.");
    DEF_GATE(ZGate, "Specific class of Pauli-Z gate.");
    DEF_GATE(HGate, "Specific class of Hadamard gate.");
    DEF_GATE(SGate,
             "Specific class of S gate, represented as $\\begin{bmatrix}\n1 & 0\\\\\n0 & "
             "i\n\\end{bmatrix}$.");
    DEF_GATE(SdagGate, "Specific class of inverse of S gate.");
    DEF_GATE(TGate,
             "Specific class of T gate, represented as $\\begin{bmatrix}\n1 & 0\\\\\n0 & "
             "e^{i\\pi/4}\n\\end{bmatrix}$.");
    DEF_GATE(TdagGate, "Specific class of inverse of T gate.");
    DEF_GATE(SqrtXGate,
             "Specific class of sqrt(X) gate, represented as $\\begin{bmatrix}\n1+i & 1-i\\\\\n1-i "
             "& 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtXdagGate, "Specific class of inverse of sqrt(X) gate.");
    DEF_GATE(SqrtYGate,
             "Specific class of sqrt(Y) gate, represented as $\\begin{bmatrix}\n1+i & -1-i "
             "\\\\\n1+i & 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtYdagGate, "Specific class of inverse of sqrt(Y) gate.");
    DEF_GATE(
        P0Gate,
        "Specific class of projection gate to $\\ket{0}$.\n\n.. note:: This gate is not unitary.");
    DEF_GATE(
        P1Gate,
        "Specific class of projection gate to $\\ket{1}$.\n\n.. note:: This gate is not unitary.");

#define DEF_ROTATION_GATE(GATE_TYPE, DESCRIPTION) \
    DEF_GATE(GATE_TYPE, DESCRIPTION)              \
        .def(                                     \
            "angle", [](const GATE_TYPE& gate) { return gate->angle(); }, "Get `angle` property.")

    DEF_ROTATION_GATE(
        RXGate,
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}X}$.");
    DEF_ROTATION_GATE(
        RYGate,
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Y}$.");
    DEF_ROTATION_GATE(
        RZGate,
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Z}$.");

    DEF_GATE(U1Gate,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\begin{bmatrix}\n1 & 0\\\\\n0 & e^{i\\lambda}\n\\end{bmatrix}$.")
        .def(
            "lambda_", [](const U1Gate& gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U2Gate,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_", [](const U2Gate& gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(U3Gate,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, "
             "represented as "
             "$\\begin{bmatrix}\n\\cos \\frac{\\theta}{2} & "
             "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\\n"
             "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
             "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}\n\\end{bmatrix}$.")
        .def(
            "theta", [](const U3Gate& gate) { return gate->theta(); }, "Get `theta` property.")
        .def(
            "phi", [](const U3Gate& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_", [](const U3Gate& gate) { return gate->lambda(); }, "Get `lambda` property.");
    DEF_GATE(SwapGate, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
