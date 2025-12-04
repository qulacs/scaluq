#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec>
class IGateImpl : public GateBase<Prec> {
public:
    IGateImpl() : GateBase<Prec>(0, 0, 0) {}

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override { j = Json{{"type", "I"}}; }
};

template <Precision Prec>
class GlobalPhaseGateImpl : public GateBase<Prec> {
protected:
    Float<Prec> _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask,
                        std::uint64_t control_value_mask,
                        Float<Prec> phase)
        : GateBase<Prec>(0, control_mask, control_value_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Prec>>(
            this->_control_mask, this->_control_value_mask, -_phase);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"phase", this->phase()}};
    }
};

template <Precision Prec>
class RotationGateBase : public GateBase<Prec> {
protected:
    Float<Prec> _angle;

public:
    RotationGateBase(std::uint64_t target_mask,
                     std::uint64_t control_mask,
                     std::uint64_t control_value_mask,
                     Float<Prec> angle)
        : GateBase<Prec>(target_mask, control_mask, control_value_mask), _angle(angle) {}

    double angle() const { return _angle; }
};

template <Precision Prec>
class XGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "X"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class YGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Y"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class ZGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Z"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class HGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "H"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SGateImpl;
template <Precision Prec>
class SdagGateImpl;
template <Precision Prec>
class TGateImpl;
template <Precision Prec>
class TdagGateImpl;
template <Precision Prec>
class SqrtXGateImpl;
template <Precision Prec>
class SqrtXdagGateImpl;
template <Precision Prec>
class SqrtYGateImpl;
template <Precision Prec>
class SqrtYdagGateImpl;

template <Precision Prec>
class SGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SdagGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class TGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class TdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SqrtXGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SqrtXdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SqrtYGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class SqrtYdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class P0GateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P0"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class P1GateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

template <Precision Prec>
class RXGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class RYGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class RZGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class U1GateImpl : public GateBase<Prec> {
    Float<Prec> _lambda;

public:
    U1GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask, control_value_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Prec>>(
            this->_target_mask, this->_control_mask, this->_control_value_mask, -_lambda);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()},
                 {"lambda", this->lambda()}};
    }
};
template <Precision Prec>
class U2GateImpl : public GateBase<Prec> {
    Float<Prec> _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask, control_value_mask),
          _phi(phi),
          _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Prec>>(
            this->_target_mask,
            this->_control_mask,
            this->_control_value_mask,
            -_lambda - static_cast<Float<Prec>>(Kokkos::numbers::pi),
            -_phi + static_cast<Float<Prec>>(Kokkos::numbers::pi));
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

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

template <Precision Prec>
class U3GateImpl : public GateBase<Prec> {
    Float<Prec> _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               std::uint64_t control_value_mask,
               Float<Prec> theta,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask, control_value_mask),
          _theta(theta),
          _phi(phi),
          _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Prec>>(this->_target_mask,
                                                        this->_control_mask,
                                                        this->_control_value_mask,
                                                        -_theta,
                                                        -_lambda,
                                                        -_phi);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

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

template <Precision Prec>
class SwapGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec, ExecutionSpace::Host>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Host>& state_vector) const override;
#ifdef SCALUQ_USE_CUDA
    void update_quantum_state(
        StateVector<Prec, ExecutionSpace::Default>& state_vector) const override;
    void update_quantum_state(
        StateVectorBatched<Prec, ExecutionSpace::Default>& state_vector) const override;
#endif  // SCALUQ_USE_CUDA

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Swap"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"control_value", this->control_value_list()}};
    }
};

}  // namespace internal

template <Precision Prec>
using IGate = internal::GatePtr<internal::IGateImpl<Prec>>;
template <Precision Prec>
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl<Prec>>;
template <Precision Prec>
using XGate = internal::GatePtr<internal::XGateImpl<Prec>>;
template <Precision Prec>
using YGate = internal::GatePtr<internal::YGateImpl<Prec>>;
template <Precision Prec>
using ZGate = internal::GatePtr<internal::ZGateImpl<Prec>>;
template <Precision Prec>
using HGate = internal::GatePtr<internal::HGateImpl<Prec>>;
template <Precision Prec>
using SGate = internal::GatePtr<internal::SGateImpl<Prec>>;
template <Precision Prec>
using SdagGate = internal::GatePtr<internal::SdagGateImpl<Prec>>;
template <Precision Prec>
using TGate = internal::GatePtr<internal::TGateImpl<Prec>>;
template <Precision Prec>
using TdagGate = internal::GatePtr<internal::TdagGateImpl<Prec>>;
template <Precision Prec>
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl<Prec>>;
template <Precision Prec>
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl<Prec>>;
template <Precision Prec>
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl<Prec>>;
template <Precision Prec>
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl<Prec>>;
template <Precision Prec>
using P0Gate = internal::GatePtr<internal::P0GateImpl<Prec>>;
template <Precision Prec>
using P1Gate = internal::GatePtr<internal::P1GateImpl<Prec>>;
template <Precision Prec>
using RXGate = internal::GatePtr<internal::RXGateImpl<Prec>>;
template <Precision Prec>
using RYGate = internal::GatePtr<internal::RYGateImpl<Prec>>;
template <Precision Prec>
using RZGate = internal::GatePtr<internal::RZGateImpl<Prec>>;
template <Precision Prec>
using U1Gate = internal::GatePtr<internal::U1GateImpl<Prec>>;
template <Precision Prec>
using U2Gate = internal::GatePtr<internal::U2GateImpl<Prec>>;
template <Precision Prec>
using U3Gate = internal::GatePtr<internal::U3GateImpl<Prec>>;
template <Precision Prec>
using SwapGate = internal::GatePtr<internal::SwapGateImpl<Prec>>;

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_standard_hpp(nb::module_& m, nb::class_<Gate<Prec>>& gate_base_def) {
    bind_specific_gate<IGate<Prec>, Prec>(m, gate_base_def, "IGate", "Specific class of I gate.");
    bind_specific_gate<GlobalPhaseGate<Prec>, Prec>(
        m,
        gate_base_def,
        "GlobalPhaseGate",
        "Specific class of gate, which rotate global phase, represented as $e^{i\\gamma}I$.");
    gate_base_def.def(
        "phase",
        [](const GlobalPhaseGate<Prec>& gate) { return gate->phase(); },
        "Get `phase` property. The phase is represented as $\\gamma$.");
    bind_specific_gate<XGate<Prec>, Prec>(
        m, gate_base_def, "XGate", "Specific class of Pauli-X gate.");
    bind_specific_gate<YGate<Prec>, Prec>(
        m, gate_base_def, "YGate", "Specific class of Pauli-Y gate.");
    bind_specific_gate<ZGate<Prec>, Prec>(
        m, gate_base_def, "ZGate", "Specific class of Pauli-Z gate.");
    bind_specific_gate<HGate<Prec>, Prec>(
        m, gate_base_def, "HGate", "Specific class of Hadamard gate.");
    bind_specific_gate<SGate<Prec>, Prec>(
        m,
        gate_base_def,
        "SGate",
        "Specific class of S gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 & i "
        "\\end{bmatrix}$.");
    bind_specific_gate<SdagGate<Prec>, Prec>(
        m, gate_base_def, "SdagGate", "Specific class of inverse of S gate.");
    bind_specific_gate<TGate<Prec>, Prec>(
        m,
        gate_base_def,
        "TGate",
        "Specific class of T gate, represented as $\\begin{bmatrix} 1 & 0 \\\\ 0 &"
        "e^{i \\pi/4} \\end{bmatrix}$.");
    bind_specific_gate<TdagGate<Prec>, Prec>(
        m, gate_base_def, "TdagGate", "Specific class of inverse of T gate.");
    bind_specific_gate<SqrtXGate<Prec>, Prec>(
        m,
        gate_base_def,
        "SqrtXGate",
        "Specific class of sqrt(X) gate, represented as $\\frac{1}{\\sqrt{2}} "
        "\\begin{bmatrix} 1+i & 1-i\\\\ 1-i "
        "& 1+i \\end{bmatrix}$.");
    bind_specific_gate<SqrtXdagGate<Prec>, Prec>(
        m,
        gate_base_def,
        "SqrtXdagGate",
        "Specific class of inverse of sqrt(X) gate, represented as "
        "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1+i\\\\ 1+i & 1-i \\end{bmatrix}$.");
    bind_specific_gate<SqrtYGate<Prec>, Prec>(
        m,
        gate_base_def,
        "SqrtYGate",
        "Specific class of sqrt(Y) gate, represented as $\\frac{1}{\\sqrt{2}} "
        "\\begin{bmatrix} 1+i & -1-i "
        "\\\\ 1+i & 1+i \\end{bmatrix}$.");
    bind_specific_gate<SqrtYdagGate<Prec>, Prec>(
        m,
        gate_base_def,
        "SqrtYdagGate",
        "Specific class of inverse of sqrt(Y) gate, represented as "
        "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix} 1-i & 1-i\\\\ -1+i & 1-i \\end{bmatrix}$.");
    bind_specific_gate<P0Gate<Prec>, Prec>(
        m,
        gate_base_def,
        "P0Gate",
        "Specific class of projection gate to $\\ket{0}$.\n\nNotes:\n\tThis gate is "
        "not unitary.");
    bind_specific_gate<P1Gate<Prec>, Prec>(
        m,
        gate_base_def,
        "P1Gate",
        "Specific class of projection gate to $\\ket{1}$.\n\nNotes:\n\tThis gate is "
        "not unitary.");
    bind_specific_gate<RXGate<Prec>, Prec>(
        m,
        gate_base_def,
        "RXGate",
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\theta}{2}X}$.")
        .def(
            "angle",
            [](const RXGate<Prec>& gate) { return gate->angle(); },
            "Get `angle` property.");
    bind_specific_gate<RYGate<Prec>, Prec>(
        m,
        gate_base_def,
        "RYGate",
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\theta}{2}Y}$.")
        .def(
            "angle",
            [](const RYGate<Prec>& gate) { return gate->angle(); },
            "Get `angle` property.");
    bind_specific_gate<RZGate<Prec>, Prec>(
        m,
        gate_base_def,
        "RZGate",
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\theta}{2}Z}$.")
        .def(
            "angle",
            [](const RZGate<Prec>& gate) { return gate->angle(); },
            "Get `angle` property.");
    bind_specific_gate<U1Gate<Prec>, Prec>(
        m,
        gate_base_def,
        "U1Gate",
        "Specific class of IBMQ's U1 Gate, which is a rotation about Z-axis, "
        "represented as "
        "$\\begin{bmatrix} 1 & 0 \\\\ 0 & e^{i\\lambda} \\end{bmatrix}$.")
        .def(
            "lambda_",
            [](const U1Gate<Prec>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    bind_specific_gate<U2Gate<Prec>, Prec>(
        m,
        gate_base_def,
        "U2Gate",
        "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
        "represented as "
        "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\ "
        "e^{i\\phi} & e^{i(\\phi+\\lambda)} \\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Prec>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    bind_specific_gate<U3Gate<Prec>, Prec>(
        m,
        gate_base_def,
        "U3Gate",
        "Specific class of IBMQ's U3 Gate, which is a rotation about 3 axis, "
        "represented as "
        "$\\begin{bmatrix} \\cos \\frac{\\theta}{2} & "
        "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\ "
        "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
        "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2} \\end{bmatrix}$.")
        .def(
            "theta",
            [](const U3Gate<Prec>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Prec>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    bind_specific_gate<SwapGate<Prec>, Prec>(
        m, gate_base_def, "SwapGate", "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
