#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
class IGateImpl : public GateBase<Prec, Space> {
public:
    IGateImpl() : GateBase<Prec, Space>(0, 0, 0) {}

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override { j = Json{{"type", "I"}}; }
};

template <Precision Prec, ExecutionSpace Space>
class GlobalPhaseGateImpl : public GateBase<Prec, Space> {
protected:
    Float<Prec> _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        Float<Prec> phase)
        : GateBase<Prec, Space>(0, control_mask, control_value_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Prec, Space>>(
            this->_control_mask, this->_control_value_mask, -_phase);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"phase", this->phase()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RotationGateBase : public GateBase<Prec, Space> {
protected:
    Float<Prec> _angle;

public:
    RotationGateBase(std::uint64_t target_mask,
                     std::uint64_t control_mask,
                     std::uint64_t control_value_mask,
                     Float<Prec> angle)
        : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask), _angle(angle) {}

    double angle() const { return _angle; }
};

template <Precision Prec, ExecutionSpace Space>
class XGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "X"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class YGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Y"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class ZGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Z"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class HGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "H"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class TGateImpl;
template <Precision Prec, ExecutionSpace Space>
class TdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtXGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtXdagGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtYGateImpl;
template <Precision Prec, ExecutionSpace Space>
class SqrtYdagGateImpl;

template <Precision Prec, ExecutionSpace Space>
class SGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SdagGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class TGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class TdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtXGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtXdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtYGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtYdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class P0GateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P0"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class P1GateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RXGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RYGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RZGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class U1GateImpl : public GateBase<Prec, Space> {
    Float<Prec> _lambda;

public:
    U1GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -_lambda);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"lambda", this->lambda()}};
    }
};
template <Precision Prec, ExecutionSpace Space>
class U2GateImpl : public GateBase<Prec, Space> {
    Float<Prec> _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask),
          _phi(phi),
          _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Prec, Space>>(
            this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            -_lambda - static_cast<Float<Prec>>(Kokkos::numbers::pi),
            -_phi + static_cast<Float<Prec>>(Kokkos::numbers::pi));
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U2"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"lambda", this->lambda()},
                 {"phi", this->phi()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class U3GateImpl : public GateBase<Prec, Space> {
    Float<Prec> _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> theta,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask, control_value_mask),
          _theta(theta),
          _phi(phi),
          _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Prec, Space>>(this->_target_mask,
                                                               this->_control_mask,
                                                               this->_control_value_mask,
                                                               -_theta,
                                                               -_lambda,
                                                               -_phi);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U3"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"lambda", this->lambda()},
                 {"phi", this->phi()},
                 {"theta", this->theta()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SwapGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Swap"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

}  // namespace internal

template <Precision Prec, ExecutionSpace Space>
using IGate = internal::GatePtr<internal::IGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using XGate = internal::GatePtr<internal::XGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using YGate = internal::GatePtr<internal::YGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using ZGate = internal::GatePtr<internal::ZGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using HGate = internal::GatePtr<internal::HGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SGate = internal::GatePtr<internal::SGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SdagGate = internal::GatePtr<internal::SdagGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using TGate = internal::GatePtr<internal::TGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using TdagGate = internal::GatePtr<internal::TdagGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using P0Gate = internal::GatePtr<internal::P0GateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using P1Gate = internal::GatePtr<internal::P1GateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using RXGate = internal::GatePtr<internal::RXGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using RYGate = internal::GatePtr<internal::RYGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using RZGate = internal::GatePtr<internal::RZGateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using U1Gate = internal::GatePtr<internal::U1GateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using U2Gate = internal::GatePtr<internal::U2GateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using U3Gate = internal::GatePtr<internal::U3GateImpl<Prec, Space>>;
template <Precision Prec, ExecutionSpace Space>
using SwapGate = internal::GatePtr<internal::SwapGateImpl<Prec, Space>>;

namespace internal {

// I
#define DECLARE_GET_FROM_JSON_I(Prec, Space)                                          \
    template <>                                                                       \
    inline std::shared_ptr<const IGateImpl<Prec, Space>> get_from_json(const Json&) { \
        return std::make_shared<const IGateImpl<Prec, Space>>();                      \
    }

// GlobalPhase
#define DECLARE_GET_FROM_JSON_GLOBAL_PHASE(Prec, Space)                                           \
    template <>                                                                                   \
    inline std::shared_ptr<const GlobalPhaseGateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();                  \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();            \
        return std::make_shared<const GlobalPhaseGateImpl<Prec, Space>>(                          \
            vector_to_mask(control_qubits),                                                       \
            vector_to_mask(control_qubits, control_values),                                       \
            static_cast<Float<Prec>>(j.at("phase").get<double>()));                               \
    }

// X, Y, Z, H, S, Sdag, T, Tdag, SqrtX, SqrtY, P0, P1
#define DECLARE_GET_FROM_JSON_SINGLE_IMPL(Impl, Prec, Space)                           \
    template <>                                                                        \
    inline std::shared_ptr<const Impl<Prec, Space>> get_from_json(const Json& j) {     \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();       \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>(); \
        return std::make_shared<const Impl<Prec, Space>>(                              \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),          \
            vector_to_mask(control_qubits),                                            \
            vector_to_mask(control_qubits, control_values));                           \
    }
#define DECLARE_GET_FROM_JSON_SINGLE(Prec, Space)                    \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(XGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(YGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(ZGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(HGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SdagGateImpl, Prec, Space)     \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(TGateImpl, Prec, Space)        \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(TdagGateImpl, Prec, Space)     \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtXGateImpl, Prec, Space)    \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtXdagGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtYGateImpl, Prec, Space)    \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(SqrtYdagGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(P0GateImpl, Prec, Space)       \
    DECLARE_GET_FROM_JSON_SINGLE_IMPL(P1GateImpl, Prec, Space)

// RX, RY, RZ
#define DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(Impl, Prec, Space)                         \
    template <>                                                                        \
    inline std::shared_ptr<const Impl<Prec, Space>> get_from_json(const Json& j) {     \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();       \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>(); \
        return std::make_shared<const Impl<Prec, Space>>(                              \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),          \
            vector_to_mask(control_qubits),                                            \
            vector_to_mask(control_qubits, control_values),                            \
            static_cast<Float<Prec>>(j.at("angle").get<double>()));                    \
    }
#define DECLARE_GET_FROM_JSON_R_SINGLE(Prec, Space)              \
    DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RXGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RYGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_R_SINGLE_IMPL(RZGateImpl, Prec, Space)

// U1, U2, U3
#define DECLARE_GET_FROM_JSON_U(Prec, Space)                                             \
    template <>                                                                          \
    inline std::shared_ptr<const U1GateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();         \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const U1GateImpl<Prec, Space>>(                          \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),            \
            vector_to_mask(control_qubits),                                              \
            vector_to_mask(control_qubits, control_values),                              \
            static_cast<Float<Prec>>(j.at("theta").get<double>()));                      \
    }                                                                                    \
    template <>                                                                          \
    inline std::shared_ptr<const U2GateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();         \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const U2GateImpl<Prec, Space>>(                          \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),            \
            vector_to_mask(control_qubits),                                              \
            vector_to_mask(control_qubits, control_values),                              \
            static_cast<Float<Prec>>(j.at("theta").get<double>()),                       \
            static_cast<Float<Prec>>(j.at("phi").get<double>()));                        \
    }                                                                                    \
    template <>                                                                          \
    inline std::shared_ptr<const U3GateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();         \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const U3GateImpl<Prec, Space>>(                          \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),            \
            vector_to_mask(control_qubits),                                              \
            vector_to_mask(control_qubits, control_values),                              \
            static_cast<Float<Prec>>(j.at("theta").get<double>()),                       \
            static_cast<Float<Prec>>(j.at("phi").get<double>()),                         \
            static_cast<Float<Prec>>(j.at("labmda").get<double>()));                     \
    }

// Swap
#define DECLARE_GET_FROM_JSON_SWAP(Prec, Space)                                            \
    template <>                                                                            \
    inline std::shared_ptr<const SwapGateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto control_qubits = j.at("control").get<std::vector<std::uint64_t>>();           \
        auto control_values = j.at("control_value").get<std::vector<std::uint64_t>>();     \
        return std::make_shared<const SwapGateImpl<Prec, Space>>(                          \
            vector_to_mask(j.at("target").get<std::vector<std::uint64_t>>()),              \
            vector_to_mask(control_qubits),                                                \
            vector_to_mask(control_qubits, control_values));                               \
    }

// Instantiate get_from_json in each gate classes
#define INSTANTIATE_GET_FROM_JSON_EACH_SPACE(Prec)                    \
    DECLARE_GET_FROM_JSON_I(Prec, ExecutionSpace::Default)            \
    DECLARE_GET_FROM_JSON_I(Prec, ExecutionSpace::Host)               \
    DECLARE_GET_FROM_JSON_GLOBAL_PHASE(Prec, ExecutionSpace::Default) \
    DECLARE_GET_FROM_JSON_GLOBAL_PHASE(Prec, ExecutionSpace::Host)    \
    DECLARE_GET_FROM_JSON_SINGLE(Prec, ExecutionSpace::Default)       \
    DECLARE_GET_FROM_JSON_SINGLE(Prec, ExecutionSpace::Host)          \
    DECLARE_GET_FROM_JSON_R_SINGLE(Prec, ExecutionSpace::Default)     \
    DECLARE_GET_FROM_JSON_R_SINGLE(Prec, ExecutionSpace::Host)        \
    DECLARE_GET_FROM_JSON_U(Prec, ExecutionSpace::Default)            \
    DECLARE_GET_FROM_JSON_U(Prec, ExecutionSpace::Host)               \
    DECLARE_GET_FROM_JSON_SWAP(Prec, ExecutionSpace::Default)         \
    DECLARE_GET_FROM_JSON_SWAP(Prec, ExecutionSpace::Host)
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

#undef DECLARE_GET_FROM_JSON_I
#undef DECLARE_GET_FROM_JSON_GLOBAL_PHASE
#undef DECLARE_GET_FROM_JSON_SINGLE_IMPL
#undef DECLARE_GET_FROM_JSON_SINGLE
#undef DECLARE_GET_FROM_JSON_R_SINGLE_IMPL
#undef DECLARE_GET_FROM_JSON_R_SINGLE
#undef DECLARE_GET_FROM_JSON_U
#undef DECLARE_GET_FROM_JSON_SWAP
#undef INSTANTIATE_GET_FROM_JSON_EACH_SPACE

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_standard_hpp(nb::module_& m, nb::class_<Gate<Prec, Space>>& gate_base_def) {
    DEF_GATE(IGate, Prec, Space, "Specific class of Pauli-I gate.", gate_base_def);
    DEF_GATE(GlobalPhaseGate,
             Prec,
             Space,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\gamma}I$.",
             gate_base_def)
        .def(
            "phase",
            [](const GlobalPhaseGate<Prec, Space>& gate) { return gate->phase(); },
            "Get `phase` property. The phase is represented as $\\gamma$.",
            gate_base_def);
    DEF_GATE(XGate, Prec, Space, "Specific class of Pauli-X gate.", gate_base_def);
    DEF_GATE(YGate, Prec, Space, "Specific class of Pauli-Y gate.", gate_base_def);
    DEF_GATE(ZGate, Prec, Space, "Specific class of Pauli-Z gate.", gate_base_def);
    DEF_GATE(HGate, Prec, Space, "Specific class of Hadamard gate.", gate_base_def);
    DEF_GATE(SGate,
             Prec,
             Space,
             "Specific class of S gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 & i "
             "\\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(SdagGate, Prec, Space, "Specific class of inverse of S gate.", gate_base_def);
    DEF_GATE(TGate,
             Prec,
             Space,
             "Specific class of T gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 &"
             "e^{i \\pi/4} \\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(TdagGate, Prec, Space, "Specific class of inverse of T gate.", gate_base_def);
    DEF_GATE(SqrtXGate,
             Prec,
             Space,
             "Specific class of sqrt(X) gate, represented as $\\frac{1}{\\sqrt{2}} "
             "\\begin{bmatrix} 1+i & 1-i\\\\ 1-i "
             "& 1+i \\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(SqrtXdagGate,
             Prec,
             Space,
             "Specific class of inverse of sqrt(X) gate, represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1+i\\\\ 1+i & 1-i \\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(SqrtYGate,
             Prec,
             Space,
             "Specific class of sqrt(Y) gate, represented as $\\frac{1}{\\sqrt{2}} "
             "\\begin{bmatrix} 1+i & -1-i "
             "\\\\ 1+i & 1+i \\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(SqrtYdagGate,
             Prec,
             Space,
             "Specific class of inverse of sqrt(Y) gate, represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1-i\\\\ -1+i & 1-i \\end{bmatrix}$.",
             gate_base_def);
    DEF_GATE(P0Gate,
             Prec,
             Space,
             "Specific class of projection gate to $\\ket{0}$.\n\nNotes:\n\tThis gate is "
             "not unitary.",
             gate_base_def);
    DEF_GATE(P1Gate,
             Prec,
             Space,
             "Specific class of projection gate to $\\ket{1}$.\n\nNotes:\n\tThis gate is "
             "not unitary.",
             gate_base_def);

#define DEF_ROTATION_GATE(GATE_TYPE, PRECISION, SPACE, DESCRIPTION, GATE_BASE_DEF) \
    DEF_GATE(GATE_TYPE, PRECISION, SPACE, DESCRIPTION, GATE_BASE_DEF)              \
        .def(                                                                      \
            "angle",                                                               \
            [](const GATE_TYPE<PRECISION, SPACE>& gate) { return gate->angle(); }, \
            "Get `angle` property.")

    DEF_ROTATION_GATE(
        RXGate,
        Prec,
        Space,
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\theta}{2}X}$.",
        gate_base_def);
    DEF_ROTATION_GATE(
        RYGate,
        Prec,
        Space,
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\theta}{2}Y}$.",
        gate_base_def);
    DEF_ROTATION_GATE(
        RZGate,
        Prec,
        Space,
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\theta}{2}Z}$.",
        gate_base_def);

    DEF_GATE(U1Gate,
             Prec,
             Space,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\begin{bmatrix} 1 & 0 \\\\ 0 & e^{i\\lambda} \\end{bmatrix}$.",
             gate_base_def)
        .def(
            "lambda_",
            [](const U1Gate<Prec, Space>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Prec,
             Space,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\ "
             "e^{i\\phi} & e^{i(\\phi+\\lambda)} \\end{bmatrix}$.",
             gate_base_def)
        .def(
            "phi",
            [](const U2Gate<Prec, Space>& gate) { return gate->phi(); },
            "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Prec, Space>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U3Gate,
             Prec,
             Space,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, "
             "represented as "
             "$\\begin{bmatrix} \\cos \\frac{\\theta}{2} & "
             "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\ "
             "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
             "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2} \\end{bmatrix}$.",
             gate_base_def)
        .def(
            "theta",
            [](const U3Gate<Prec, Space>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi",
            [](const U3Gate<Prec, Space>& gate) { return gate->phi(); },
            "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Prec, Space>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(SwapGate, Prec, Space, "Specific class of two-qubit swap gate.", gate_base_def);
}
}  // namespace internal
#endif
}  // namespace scaluq
