#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point FloatType>
class IGateImpl : public GateBase<FloatType> {
public:
    IGateImpl() : GateBase<FloatType>(0, 0) {}

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        return internal::ComplexMatrix::Identity(1, 1);
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        i_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: I\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class GlobalPhaseGateImpl : public GateBase<FloatType> {
protected:
    double _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask, double phase)
        : GateBase<FloatType>(0, control_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<FloatType>>(this->_control_mask, -_phase);
    }
    internal::ComplexMatrix get_matrix() const override {
        return internal::ComplexMatrix::Identity(1, 1) * std::exp(std::complex<double>(0, _phase));
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        global_phase_gate(this->_target_mask, this->_control_mask, _phase, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: GlobalPhase\n";
        ss << indent << "  Phase: " << _phase << "\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class RotationGateBase : public GateBase<FloatType> {
protected:
    double _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, double angle)
        : GateBase<FloatType>(target_mask, control_mask), _angle(angle) {}

    double angle() const { return _angle; }
};

template <std::floating_point FloatType>
class XGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, 1, 1, 0;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        x_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: X\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class YGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, -1i, 1i, 0;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        y_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Y\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class ZGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        z_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Z\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class HGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 1, 1, -1;
        mat /= std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        h_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: H\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SGateImpl;
template <std::floating_point FloatType>
class SdagGateImpl;
template <std::floating_point FloatType>
class TGateImpl;
template <std::floating_point FloatType>
class TdagGateImpl;
template <std::floating_point FloatType>
class SqrtXGateImpl;
template <std::floating_point FloatType>
class SqrtXdagGateImpl;
template <std::floating_point FloatType>
class SqrtYGateImpl;
template <std::floating_point FloatType>
class SqrtYdagGateImpl;

template <std::floating_point FloatType>
class SGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SdagGateImpl<FloatType>>(this->_target_mask,
                                                               this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 1i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        s_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: S\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SdagGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SGateImpl<FloatType>>(this->_target_mask,
                                                            this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sdag_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Sdag\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class TGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<FloatType>>(this->_target_mask,
                                                               this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. + 1i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        t_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: T\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class TdagGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const TGateImpl<FloatType>>(this->_target_mask,
                                                            this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, (1. - 1.i) / std::sqrt(2);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        tdag_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Tdag\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SqrtXGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<FloatType>>(this->_target_mask,
                                                                   this->_control_mask);
    }

    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sqrtx_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtX\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SqrtXdagGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<FloatType>>(this->_target_mask,
                                                                this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sqrtxdag_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtXdag\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SqrtYGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<FloatType>>(this->_target_mask,
                                                                   this->_control_mask);
    }

    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sqrty_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtY\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SqrtYdagGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<FloatType>>(this->_target_mask,
                                                                this->_control_mask);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i, 0.5 - 0.5i;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        sqrtydag_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: SqrtYdag\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class P0GateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, 0;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        p0_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: P0\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class P1GateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        p1_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: P1\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class RXGateImpl : public RotationGateBase<FloatType> {
public:
    using RotationGateBase<FloatType>::RotationGateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<FloatType>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(this->_angle / 2), -1i * std::sin(this->_angle / 2),
            -1i * std::sin(this->_angle / 2), std::cos(this->_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        rx_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RX\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class RYGateImpl : public RotationGateBase<FloatType> {
public:
    using RotationGateBase<FloatType>::RotationGateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<FloatType>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(this->_angle / 2), -std::sin(this->_angle / 2), std::sin(this->_angle / 2),
            std::cos(this->_angle / 2);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        ry_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RY\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class RZGateImpl : public RotationGateBase<FloatType> {
public:
    using RotationGateBase<FloatType>::RotationGateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<FloatType>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::exp(-0.5i * this->_angle), 0, 0, std::exp(0.5i * this->_angle);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        rz_gate(this->_target_mask, this->_control_mask, this->_angle, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: RZ\n";
        ss << indent << "  Angle: " << this->_angle << "\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class U1GateImpl : public GateBase<FloatType> {
    double _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, double lambda)
        : GateBase<FloatType>(target_mask, control_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<FloatType>>(
            this->_target_mask, this->_control_mask, -_lambda);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, std::exp(1i * _lambda);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        u1_gate(this->_target_mask, this->_control_mask, _lambda, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U1\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};
template <std::floating_point FloatType>
class U2GateImpl : public GateBase<FloatType> {
    double _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, double phi, double lambda)
        : GateBase<FloatType>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<FloatType>>(this->_target_mask,
                                                             this->_control_mask,
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

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        u2_gate(this->_target_mask, this->_control_mask, _phi, _lambda, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U2\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class U3GateImpl : public GateBase<FloatType> {
    double _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               double theta,
               double phi,
               double lambda)
        : GateBase<FloatType>(target_mask, control_mask),
          _theta(theta),
          _phi(phi),
          _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<FloatType>>(
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat(2, 2);
        mat << std::cos(_theta / 2.), -std::exp(1i * _lambda) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::sin(_theta / 2.),
            std::exp(1i * _phi) * std::exp(1i * _lambda) * std::cos(_theta / 2.);
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        u3_gate(this->_target_mask, this->_control_mask, _theta, _phi, _lambda, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: U3\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

template <std::floating_point FloatType>
class SwapGateImpl : public GateBase<FloatType> {
public:
    using GateBase<FloatType>::GateBase;

    std::shared_ptr<const GateBase<FloatType>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix get_matrix() const override {
        internal::ComplexMatrix mat = internal::ComplexMatrix::Identity(1 << 2, 1 << 2);
        mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
        return mat;
    }

    void update_quantum_state(StateVector<FloatType>& state_vector) const override {
        this->check_qubit_mask_within_bounds(state_vector);
        swap_gate(this->_target_mask, this->_control_mask, state_vector);
    }

    std::string to_string(const std::string& indent) const override {
        std::ostringstream ss;
        ss << indent << "Gate Type: Swap\n";
        ss << this->get_qubit_info_as_string(indent);
        return ss.str();
    }
};

}  // namespace internal

template <std::floating_point FloatType>
using IGate = internal::GatePtr<internal::IGateImpl<FloatType>>;
template <std::floating_point FloatType>
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl<FloatType>>;
template <std::floating_point FloatType>
using XGate = internal::GatePtr<internal::XGateImpl<FloatType>>;
template <std::floating_point FloatType>
using YGate = internal::GatePtr<internal::YGateImpl<FloatType>>;
template <std::floating_point FloatType>
using ZGate = internal::GatePtr<internal::ZGateImpl<FloatType>>;
template <std::floating_point FloatType>
using HGate = internal::GatePtr<internal::HGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SGate = internal::GatePtr<internal::SGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SdagGate = internal::GatePtr<internal::SdagGateImpl<FloatType>>;
template <std::floating_point FloatType>
using TGate = internal::GatePtr<internal::TGateImpl<FloatType>>;
template <std::floating_point FloatType>
using TdagGate = internal::GatePtr<internal::TdagGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl<FloatType>>;
template <std::floating_point FloatType>
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl<FloatType>>;
template <std::floating_point FloatType>
using P0Gate = internal::GatePtr<internal::P0GateImpl<FloatType>>;
template <std::floating_point FloatType>
using P1Gate = internal::GatePtr<internal::P1GateImpl<FloatType>>;
template <std::floating_point FloatType>
using RXGate = internal::GatePtr<internal::RXGateImpl<FloatType>>;
template <std::floating_point FloatType>
using RYGate = internal::GatePtr<internal::RYGateImpl<FloatType>>;
template <std::floating_point FloatType>
using RZGate = internal::GatePtr<internal::RZGateImpl<FloatType>>;
template <std::floating_point FloatType>
using U1Gate = internal::GatePtr<internal::U1GateImpl<FloatType>>;
template <std::floating_point FloatType>
using U2Gate = internal::GatePtr<internal::U2GateImpl<FloatType>>;
template <std::floating_point FloatType>
using U3Gate = internal::GatePtr<internal::U3GateImpl<FloatType>>;
template <std::floating_point FloatType>
using SwapGate = internal::GatePtr<internal::SwapGateImpl<FloatType>>;

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
             "Specific class of S gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "i\n\\end{bmatrix}$.");
    DEF_GATE(SdagGate, "Specific class of inverse of S gate.");
    DEF_GATE(TGate,
             "Specific class of T gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "e^{i\\pi/4}\n\\end{bmatrix}$.");
    DEF_GATE(TdagGate, "Specific class of inverse of T gate.");
    DEF_GATE(
        SqrtXGate,
        "Specific class of sqrt(X) gate, represented as $\\begin{ bmatrix }\n1+i & 1-i\\\\\n1-i "
        "& 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtXdagGate, "Specific class of inverse of sqrt(X) gate.");
    DEF_GATE(SqrtYGate,
             "Specific class of sqrt(Y) gate, represented as $\\begin{ bmatrix }\n1+i & -1-i "
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
