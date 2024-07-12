#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {
class OneQubitGateBase : public GateBase {
protected:
    UINT _target;

public:
    OneQubitGateBase(UINT target) : _target(target){};

    UINT target() const { return _target; }

    std::vector<UINT> get_target_qubit_list() const override { return {_target}; }
    std::vector<UINT> get_control_qubit_list() const override { return {}; };
};

class OneQubitRotationGateBase : public OneQubitGateBase {
protected:
    double _angle;

public:
    OneQubitRotationGateBase(UINT target, double angle) : OneQubitGateBase(target), _angle(angle){};

    double angle() const { return _angle; }
};

class XGateImpl : public OneQubitGateBase {
public:
    XGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        return std::const_pointer_cast<GateBase>(shared_from_this());
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, 1, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        x_gate(this->_target, state_vector);
    }
};

class YGateImpl : public OneQubitGateBase {
public:
    YGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        return std::const_pointer_cast<GateBase>(shared_from_this());
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, -1i, 1i, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        y_gate(this->_target, state_vector);
    }
};

class ZGateImpl : public OneQubitGateBase {
public:
    ZGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        return std::const_pointer_cast<GateBase>(shared_from_this());
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        z_gate(this->_target, state_vector);
    }
};

class HGateImpl : public OneQubitGateBase {
public:
    HGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        return std::const_pointer_cast<GateBase>(shared_from_this());
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 1, 1, -1;
        mat /= std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        h_gate(this->_target, state_vector);
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

class SGateImpl : public OneQubitGateBase {
public:
    SGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        s_gate(this->_target, state_vector);
    }
};

class SdagGateImpl : public OneQubitGateBase {
public:
    SdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override { return std::make_shared<SGateImpl>(_target); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        sdag_gate(this->_target, state_vector);
    }
};
// for resolving dependency issues
inline Gate SGateImpl::get_inverse() const { return std::make_shared<SdagGateImpl>(_target); }

class TGateImpl : public OneQubitGateBase {
public:
    TGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. + 1i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        t_gate(this->_target, state_vector);
    }
};

class TdagGateImpl : public OneQubitGateBase {
public:
    TdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override { return std::make_shared<TGateImpl>(_target); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. - 1.i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        tdag_gate(this->_target, state_vector);
    }
};
// for resolving dependency issues
inline Gate TGateImpl::get_inverse() const { return std::make_shared<TdagGateImpl>(_target); }

class SqrtXGateImpl : public OneQubitGateBase {
public:
    SqrtXGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        sqrtx_gate(this->_target, state_vector);
    }
};

class SqrtXdagGateImpl : public OneQubitGateBase {
public:
    SqrtXdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override { return std::make_shared<SqrtXGateImpl>(_target); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        sqrtxdag_gate(this->_target, state_vector);
    }
};
// for resolving dependency issues
inline Gate SqrtXGateImpl::get_inverse() const {
    return std::make_shared<SqrtXdagGateImpl>(_target);
}

class SqrtYGateImpl : public OneQubitGateBase {
public:
    SqrtYGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override;
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        sqrty_gate(this->_target, state_vector);
    }
};

class SqrtYdagGateImpl : public OneQubitGateBase {
public:
    SqrtYdagGateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override { return std::make_shared<SqrtYGateImpl>(_target); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        sqrtydag_gate(this->_target, state_vector);
    }
};
// for resolving dependency issues
inline Gate SqrtYGateImpl::get_inverse() const {
    return std::make_shared<SqrtYdagGateImpl>(_target);
}

class P0GateImpl : public OneQubitGateBase {
public:
    P0GateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        p0_gate(this->_target, state_vector);
    }
};

class P1GateImpl : public OneQubitGateBase {
public:
    P1GateImpl(UINT target) : OneQubitGateBase(target){};

    Gate get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        p1_gate(this->_target, state_vector);
    }
};

class RXGateImpl : public OneQubitRotationGateBase {
public:
    RXGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate get_inverse() const override { return std::make_shared<RXGateImpl>(_target, -_angle); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -1i * std::sin(_angle / 2), -1i * std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        rx_gate(this->_target, this->_angle, state_vector);
    }
};

class RYGateImpl : public OneQubitRotationGateBase {
public:
    RYGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate get_inverse() const override { return std::make_shared<RYGateImpl>(_target, -_angle); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -std::sin(_angle / 2), std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        ry_gate(this->_target, this->_angle, state_vector);
    }
};

class RZGateImpl : public OneQubitRotationGateBase {
public:
    RZGateImpl(UINT target, double angle) : OneQubitRotationGateBase(target, angle){};

    Gate get_inverse() const override { return std::make_shared<RZGateImpl>(_target, -_angle); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * _angle), 0, 0, std::exp(0.5i * _angle);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        rz_gate(this->_target, this->_angle, state_vector);
    }
};

class U1GateImpl : public OneQubitGateBase {
    double _lambda;

public:
    U1GateImpl(UINT target, double lambda) : OneQubitGateBase(target), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    Gate get_inverse() const override { return std::make_shared<U1GateImpl>(_target, -_lambda); }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, std::exp(1i * _lambda);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        u1_gate(this->_target, this->_lambda, state_vector);
    }
};
class U2GateImpl : public OneQubitGateBase {
    double _phi, _lambda;

public:
    U2GateImpl(UINT target, double phi, double lambda)
        : OneQubitGateBase(target), _phi(phi), _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate get_inverse() const override {
        return std::make_shared<U2GateImpl>(_target, -_lambda - PI(), -_phi + PI());
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(PI() / 4.), -std::exp(1i * _lambda) * std::sin(PI() / 4.),
            std::exp(1i * _phi) * std::sin(PI() / 4.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(PI() / 4.);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        u2_gate(this->_target, this->_phi, this->_lambda, state_vector);
    }
};

class U3GateImpl : public OneQubitGateBase {
    double _theta, _phi, _lambda;

public:
    U3GateImpl(UINT target, double theta, double phi, double lambda)
        : OneQubitGateBase(target), _theta(theta), _phi(phi), _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    Gate get_inverse() const override {
        return std::make_shared<U3GateImpl>(_target, -_theta, -_lambda, -_phi);
    }
    std::optional<ComplexMatrix> get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_theta / 2.), -std::exp(1i * _lambda) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(_theta / 2.);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_within_bounds(state_vector, this->_target);
        u3_gate(this->_target, this->_theta, this->_phi, this->_lambda, state_vector);
    }
};

}  // namespace internal

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
}  // namespace scaluq
