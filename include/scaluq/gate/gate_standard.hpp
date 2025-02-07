#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec, ExecutionSpace Space>
class IGateImpl : public GateBase<Prec, Space> {
public:
    IGateImpl() : GateBase<Prec, Space>(0, 0) {}

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
    GlobalPhaseGateImpl(std::uint64_t control_mask, Float<Prec> phase)
        : GateBase<Prec, Space>(0, control_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Prec, Space>>(this->_control_mask,
                                                                        -_phase);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"phase", this->phase()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RotationGateBase : public GateBase<Prec, Space> {
protected:
    Float<Prec> _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> angle)
        : GateBase<Prec, Space>(target_mask, control_mask), _angle(angle) {}

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
                 {"control", this->control_qubit_list()}};
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
                 {"control", this->control_qubit_list()}};
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
                 {"control", this->control_qubit_list()}};
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
                 {"control", this->control_qubit_list()}};
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
        return std::make_shared<const SdagGateImpl<Prec, Space>>(this->_target_mask,
                                                                 this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Prec, Space>>(this->_target_mask,
                                                              this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class TGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Prec, Space>>(this->_target_mask,
                                                                 this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class TdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Prec, Space>>(this->_target_mask,
                                                              this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtXGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Prec, Space>>(this->_target_mask,
                                                                     this->_control_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtXdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Prec, Space>>(this->_target_mask,
                                                                  this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtYGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Prec, Space>>(this->_target_mask,
                                                                     this->_control_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class SqrtYdagGateImpl : public GateBase<Prec, Space> {
public:
    using GateBase<Prec, Space>::GateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Prec, Space>>(this->_target_mask,
                                                                  this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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
                 {"control", this->control_qubit_list()}};
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
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RXGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RYGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class RZGateImpl : public RotationGateBase<Prec, Space> {
public:
    using RotationGateBase<Prec, Space>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec, ExecutionSpace Space>
class U1GateImpl : public GateBase<Prec, Space> {
    Float<Prec> _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -_lambda);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()}};
    }
};
template <Precision Prec, ExecutionSpace Space>
class U2GateImpl : public GateBase<Prec, Space> {
    Float<Prec> _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Prec, Space>>(
            this->_target_mask,
            this->_control_mask,
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
               Float<Prec> theta,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec, Space>(target_mask, control_mask),
          _theta(theta),
          _phi(phi),
          _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec, Space>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Prec, Space>>(
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, Space>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec, Space>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U3"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
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
                 {"control", this->control_qubit_list()}};
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

#define DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec, Space)   \
    template <>                                                                       \
    inline std::shared_ptr<const IGateImpl<Prec, Space>> get_from_json(const Json&) { \
        return std::make_shared<const IGateImpl<Prec, Space>>();                      \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16, scaluq::HostSpace)
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16, scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32, scaluq::HostSpace)
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32, scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64, scaluq::HostSpace)
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64, scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16, scaluq::HostSpace)
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                               scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION_AND_EXECUTION_SPACE

#define DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec, Space)     \
    template <>                                                                                   \
    inline std::shared_ptr<const GlobalPhaseGateImpl<Prec, Space>> get_from_json(const Json& j) { \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        double phase = j.at("phase").get<double>();                                               \
        return std::make_shared<const GlobalPhaseGateImpl<Prec, Space>>(                          \
            vector_to_mask(controls), static_cast<Float<Prec>>(phase));                           \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                         scaluq::HostSpace)
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                         scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                         scaluq::HostSpace)
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                         scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                         scaluq::HostSpace)
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                         scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                         scaluq::HostSpace)
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                         scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION_AND_EXECUTION_SPACE

#define DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(  \
    Impl, Prec, Space)                                                              \
    template <>                                                                     \
    inline std::shared_ptr<const Impl<Prec, Space>> get_from_json(const Json& j) {  \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();            \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();          \
        return std::make_shared<const Impl<Prec, Space>>(vector_to_mask(targets),   \
                                                         vector_to_mask(controls)); \
    }
#define DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec,  \
                                                                                       Space) \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        XGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        YGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        ZGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        HGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SdagGateImpl, Prec, Space)                                                            \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        TGateImpl, Prec, Space)                                                               \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        TdagGateImpl, Prec, Space)                                                            \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SqrtXGateImpl, Prec, Space)                                                           \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SqrtXdagGateImpl, Prec, Space)                                                        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SqrtYGateImpl, Prec, Space)                                                           \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        SqrtYdagGateImpl, Prec, Space)                                                        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        P0GateImpl, Prec, Space)                                                              \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(                \
        P1GateImpl, Prec, Space)
#ifdef SCALUQ_FLOAT16
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                               scaluq::HostSpace)
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                               scaluq::HostSpace)
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                               scaluq::HostSpace)
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                               scaluq::HostSpace)
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                               scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE
#undef DECLARE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION_AND_EXECUTION_SPACE

#define DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Impl, Prec, Space)        \
    template <>                                                                                  \
    inline std::shared_ptr<const Impl<Prec, Space>> get_from_json(const Json& j) {               \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                       \
        double angle = j.at("angle").get<double>();                                              \
        return std::make_shared<const Impl<Prec, Space>>(                                        \
            vector_to_mask(targets), vector_to_mask(controls), static_cast<Float<Prec>>(angle)); \
    }
#define DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Prec, Space)    \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(RXGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(RYGateImpl, Prec, Space) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(RZGateImpl, Prec, Space)
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                    scaluq::HostSpace)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F16,
                                                                    scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                    scaluq::HostSpace)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F32,
                                                                    scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                    scaluq::HostSpace)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::F64,
                                                                    scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                    scaluq::HostSpace)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Precision::BF16,
                                                                    scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE
#undef DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION_AND_EXECUTION_SPACE

#define DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Impl, Prec, Space)         \
    template <>                                                                                   \
    inline std::shared_ptr<const U1GateImpl<Prec, Space>> get_from_json(const Json& j) {          \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                          \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        double theta = j.at("theta").get<double>();                                               \
        return std::make_shared<const U1GateImpl<Prec, Space>>(                                   \
            vector_to_mask(targets), vector_to_mask(controls), static_cast<Float<Prec>>(theta));  \
    }                                                                                             \
    template <>                                                                                   \
    inline std::shared_ptr<const U2GateImpl<Prec, Space>> get_from_json(const Json& j) {          \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                          \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        double theta = j.at("theta").get<double>();                                               \
        double phi = j.at("phi").get<double>();                                                   \
        return std::make_shared<const U2GateImpl<Prec, Space>>(vector_to_mask(targets),           \
                                                               vector_to_mask(controls),          \
                                                               static_cast<Float<Prec>>(theta),   \
                                                               static_cast<Float<Prec>>(phi));    \
    }                                                                                             \
    template <>                                                                                   \
    inline std::shared_ptr<const U3GateImpl<Prec, Space>> get_from_json(const Json& j) {          \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                          \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                        \
        double theta = j.at("theta").get<double>();                                               \
        double phi = j.at("phi").get<double>();                                                   \
        double lambda = j.at("lambda").get<double>();                                             \
        return std::make_shared<const U3GateImpl<Prec, Space>>(vector_to_mask(targets),           \
                                                               vector_to_mask(controls),          \
                                                               static_cast<Float<Prec>>(theta),   \
                                                               static_cast<Float<Prec>>(phi),     \
                                                               static_cast<Float<Prec>>(lambda)); \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F16,
                                                               scaluq::HostSpace)
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F16,
                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F32,
                                                               scaluq::HostSpace)
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F32,
                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F64,
                                                               scaluq::HostSpace)
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::F64,
                                                               scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::BF16,
                                                               scaluq::HostSpace)
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE(U1GateImpl,
                                                               Precision::BF16,
                                                               scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION_AND_EXECUTION_SPACE

#define DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(Impl, Prec, Space) \
    template <>                                                                              \
    inline std::shared_ptr<const SwapGateImpl<Prec, Space>> get_from_json(const Json& j) {   \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                     \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                   \
        return std::make_shared<const SwapGateImpl<Prec, Space>>(vector_to_mask(targets),    \
                                                                 vector_to_mask(controls));  \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F16,
                                                                  scaluq::HostSpace)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F16,
                                                                  scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F32,
                                                                  scaluq::HostSpace)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F32,
                                                                  scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F64,
                                                                  scaluq::HostSpace)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::F64,
                                                                  scaluq::DefaultSpace)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::BF16,
                                                                  scaluq::HostSpace)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE(SwapGateImpl,
                                                                  Precision::BF16,
                                                                  scaluq::DefaultSpace)
#endif
#undef DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION_AND_EXECUTION_SPACE
}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec, ExecutionSpace Space>
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, Prec, Space, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             Prec,
             Space,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate<Prec, Space>& gate) { return gate->phase(); },
            "Get `phase` property");
    DEF_GATE(XGate, Prec, Space, "Specific class of Pauli-X gate.");
    DEF_GATE(YGate, Prec, Space, "Specific class of Pauli-Y gate.");
    DEF_GATE(ZGate, Prec, Space, "Specific class of Pauli-Z gate.");
    DEF_GATE(HGate, Prec, Space, "Specific class of Hadamard gate.");
    DEF_GATE(SGate,
             Prec,
             Space,
             "Specific class of S gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "i\n\\end{bmatrix}$.");
    DEF_GATE(SdagGate, Prec, Space, "Specific class of inverse of S gate.");
    DEF_GATE(TGate,
             Prec,
             Space,
             "Specific class of T gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "e^{i\\pi/4}\n\\end{bmatrix}$.");
    DEF_GATE(TdagGate, Prec, Space, "Specific class of inverse of T gate.");
    DEF_GATE(
        SqrtXGate,
        Prec,
        Space,
        "Specific class of sqrt(X) gate, represented as $\\begin{ bmatrix }\n1+i & 1-i\\\\\n1-i "
        "& 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtXdagGate, Prec, Space, "Specific class of inverse of sqrt(X) gate.");
    DEF_GATE(SqrtYGate,
             Prec,
             Space,
             "Specific class of sqrt(Y) gate, represented as $\\begin{ bmatrix }\n1+i & -1-i "
             "\\\\\n1+i & 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtYdagGate, Prec, Space, "Specific class of inverse of sqrt(Y) gate.");
    DEF_GATE(
        P0Gate,
        Prec,
        Space,
        "Specific class of projection gate to $\\ket{0}$.\n\n.. note:: This gate is not unitary.");
    DEF_GATE(
        P1Gate,
        Prec,
        Space,
        "Specific class of projection gate to $\\ket{1}$.\n\n.. note:: This gate is not unitary.");

#define DEF_ROTATION_GATE(GATE_TYPE, PRECISION, SPACE, DESCRIPTION)                \
    DEF_GATE(GATE_TYPE, PRECISION, SPACE, DESCRIPTION)                             \
        .def(                                                                      \
            "angle",                                                               \
            [](const GATE_TYPE<PRECISION, SPACE>& gate) { return gate->angle(); }, \
            "Get `angle` property.")

    DEF_ROTATION_GATE(
        RXGate,
        Prec,
        Space,
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}X}$.");
    DEF_ROTATION_GATE(
        RYGate,
        Prec,
        Space,
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Y}$.");
    DEF_ROTATION_GATE(
        RZGate,
        Prec,
        Space,
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Z}$.");

    DEF_GATE(U1Gate,
             Prec,
             Space,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\begin{bmatrix}\n1 & 0\\\\\n0 & e^{i\\lambda}\n\\end{bmatrix}$.")
        .def(
            "lambda_",
            [](const U1Gate<Prec, Space>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Prec,
             Space,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
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
             "$\\begin{bmatrix}\n\\cos \\frac{\\theta}{2} & "
             "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\\n"
             "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
             "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}\n\\end{bmatrix}$.")
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
    DEF_GATE(SwapGate, Prec, Space, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
