#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

class IGateImpl : public GateBase {
public:
    IGateImpl() : GateBase(0, 0) {}

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override { return ComplexMatrix::Identity(1, 1); }

    void update_quantum_state(StateVector& state_vector) const override {
        i_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        return ComplexMatrix::Identity(1, 1) * std::exp(std::complex<double>(0, _phase));
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        global_phase_gate(_target_mask, _control_mask, _phase, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, 1, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        x_gate(_target_mask, _control_mask, state_vector);
    }
};

class YGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, -1i, 1i, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        y_gate(_target_mask, _control_mask, state_vector);
    }
};

class ZGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        z_gate(_target_mask, _control_mask, state_vector);
    }
};

class HGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 1, 1, -1;
        mat /= std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        h_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        s_gate(_target_mask, _control_mask, state_vector);
    }
};

class SdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SGateImpl>(_target_mask, _control_mask);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sdag_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. + 1i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        t_gate(_target_mask, _control_mask, state_vector);
    }
};

class TdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const TGateImpl>(_target_mask, _control_mask);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. - 1.i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        tdag_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtx_gate(_target_mask, _control_mask, state_vector);
    }
};

class SqrtXdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl>(_target_mask, _control_mask);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtxdag_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrty_gate(_target_mask, _control_mask, state_vector);
    }
};

class SqrtYdagGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl>(_target_mask, _control_mask);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        sqrtydag_gate(_target_mask, _control_mask, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 0;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        p0_gate(_target_mask, _control_mask, state_vector);
    }
};

class P1GateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        p1_gate(_target_mask, _control_mask, state_vector);
    }
};

class RXGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RXGateImpl>(_target_mask, _control_mask, -_angle);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -1i * std::sin(_angle / 2), -1i * std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        rx_gate(_target_mask, _control_mask, _angle, state_vector);
    }
};

class RYGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RYGateImpl>(_target_mask, _control_mask, -_angle);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_angle / 2), -std::sin(_angle / 2), std::sin(_angle / 2),
            std::cos(_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        ry_gate(_target_mask, _control_mask, _angle, state_vector);
    }
};

class RZGateImpl : public RotationGateBase {
public:
    using RotationGateBase::RotationGateBase;

    Gate get_inverse() const override {
        return std::make_shared<const RZGateImpl>(_target_mask, _control_mask, -_angle);
    }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * _angle), 0, 0, std::exp(0.5i * _angle);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        rz_gate(_target_mask, _control_mask, _angle, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, std::exp(1i * _lambda);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        u1_gate(_target_mask, _control_mask, _lambda, state_vector);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
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
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat(2, 2);
        mat << std::cos(_theta / 2.), -std::exp(1i * _lambda) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(_theta / 2.);
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        u3_gate(_target_mask, _control_mask, _theta, _phi, _lambda, state_vector);
    }
};

class SwapGateImpl : public GateBase {
public:
    using GateBase::GateBase;

    Gate get_inverse() const override { return shared_from_this(); }
    ComplexMatrix get_matrix() const override {
        ComplexMatrix mat = ComplexMatrix::Identity(1 << 2, 1 << 2);
        mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector& state_vector) const override {
        check_qubit_mask_within_bounds(state_vector);
        swap_gate(_target_mask, _control_mask, state_vector);
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
}  // namespace scaluq
