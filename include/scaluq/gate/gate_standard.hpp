#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <std::floating_point Fp, ExecutionSpace Sp>
class IGateImpl : public GateBase<Fp, Sp> {
public:
    IGateImpl() : GateBase<Fp, Sp>(0, 0) {}

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override { j = Json{{"type", "I"}}; }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class GlobalPhaseGateImpl : public GateBase<Fp, Sp> {
protected:
    Fp _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask, Fp phase)
        : GateBase<Fp, Sp>(0, control_mask), _phase(phase){};

    [[nodiscard]] Fp phase() const { return _phase; }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Fp, Sp>>(this->_control_mask, -_phase);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"phase", this->phase()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class RotationGateBase : public GateBase<Fp, Sp> {
protected:
    Fp _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Fp angle)
        : GateBase<Fp, Sp>(target_mask, control_mask), _angle(angle) {}

    Fp angle() const { return _angle; }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class XGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "X"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class YGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Y"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class ZGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Z"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class HGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "H"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class SdagGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class TGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class TdagGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXdagGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYGateImpl;
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYdagGateImpl;

template <std::floating_point Fp, ExecutionSpace Sp>
class SGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                            this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Fp, Sp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class TGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                            this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class TdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Fp, Sp>>(this->_target_mask, this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                                this->_control_mask);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Fp, Sp>>(this->_target_mask,
                                                             this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                                this->_control_mask);
    }

    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Fp, Sp>>(this->_target_mask,
                                                             this->_control_mask);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class P0GateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P0"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class P1GateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class RXGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Fp, Sp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class RYGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Fp, Sp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class RZGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Fp, Sp>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class U1GateImpl : public GateBase<Fp, Sp> {
    Fp _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _lambda(lambda) {}

    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Fp, Sp>>(
            this->_target_mask, this->_control_mask, -_lambda);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()}};
    }
};
template <std::floating_point Fp, ExecutionSpace Sp>
class U2GateImpl : public GateBase<Fp, Sp> {
    Fp _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp phi, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    Fp phi() const { return _phi; }
    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Fp, Sp>>(this->_target_mask,
                                                          this->_control_mask,
                                                          -_lambda - Kokkos::numbers::pi,
                                                          -_phi + Kokkos::numbers::pi);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U2"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()},
                 {"phi", this->phi()}};
    }
};

template <std::floating_point Fp, ExecutionSpace Sp>
class U3GateImpl : public GateBase<Fp, Sp> {
    Fp _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp theta, Fp phi, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}

    Fp theta() const { return _theta; }
    Fp phi() const { return _phi; }
    Fp lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Fp, Sp>>(
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

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

template <std::floating_point Fp, ExecutionSpace Sp>
class SwapGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return this->shared_from_this();
    }
    internal::ComplexMatrix<Fp> get_matrix() const override;

    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Swap"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

}  // namespace internal

template <std::floating_point Fp, ExecutionSpace Sp>
using IGate = internal::GatePtr<internal::IGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using GlobalPhaseGate = internal::GatePtr<internal::GlobalPhaseGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using XGate = internal::GatePtr<internal::XGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using YGate = internal::GatePtr<internal::YGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using ZGate = internal::GatePtr<internal::ZGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using HGate = internal::GatePtr<internal::HGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SGate = internal::GatePtr<internal::SGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SdagGate = internal::GatePtr<internal::SdagGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using TGate = internal::GatePtr<internal::TGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using TdagGate = internal::GatePtr<internal::TdagGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SqrtXGate = internal::GatePtr<internal::SqrtXGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SqrtXdagGate = internal::GatePtr<internal::SqrtXdagGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SqrtYGate = internal::GatePtr<internal::SqrtYGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SqrtYdagGate = internal::GatePtr<internal::SqrtYdagGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using P0Gate = internal::GatePtr<internal::P0GateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using P1Gate = internal::GatePtr<internal::P1GateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using RXGate = internal::GatePtr<internal::RXGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using RYGate = internal::GatePtr<internal::RYGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using RZGate = internal::GatePtr<internal::RZGateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using U1Gate = internal::GatePtr<internal::U1GateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using U2Gate = internal::GatePtr<internal::U2GateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using U3Gate = internal::GatePtr<internal::U3GateImpl<Fp, Sp>>;
template <std::floating_point Fp, ExecutionSpace Sp>
using SwapGate = internal::GatePtr<internal::SwapGateImpl<Fp, Sp>>;

namespace internal {

/*#define DECLARE_GET_FROM_JSON_IGATE_WITH_TYPE(Type)                            \
    template <>                                                                \
    inline std::shared_ptr<const IGateImpl<Type>> get_from_json(const Json&) { \
        return std::make_shared<const IGateImpl<Type>>();                      \
    }
DECLARE_GET_FROM_JSON_IGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_IGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_IGATE_WITH_TYPE

#define DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_TYPE(Type)                                      \
    template <>                                                                                    \
    inline std::shared_ptr<const GlobalPhaseGateImpl<Type>> get_from_json(const Json& j) {         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                         \
        Type phase = j.at("phase").get<Type>();                                                    \
        return std::make_shared<const GlobalPhaseGateImpl<Type>>(vector_to_mask(controls), phase); \
    }
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_TYPE

#define DECLARE_GET_FROM_JSON_WITH_TYPE(Impl, Type)                          \
    template <>                                                              \
    inline std::shared_ptr<const Impl<Type>> get_from_json(const Json& j) {  \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();     \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const Impl<Type>>(vector_to_mask(targets),   \
                                                  vector_to_mask(controls)); \
    }
#define DECLARE_GET_FROM_JSON(Impl)               \
    DECLARE_GET_FROM_JSON_WITH_TYPE(Impl, double) \
    DECLARE_GET_FROM_JSON_WITH_TYPE(Impl, float)
DECLARE_GET_FROM_JSON(XGateImpl)
DECLARE_GET_FROM_JSON(YGateImpl)
DECLARE_GET_FROM_JSON(ZGateImpl)
DECLARE_GET_FROM_JSON(HGateImpl)
DECLARE_GET_FROM_JSON(SGateImpl)
DECLARE_GET_FROM_JSON(SdagGateImpl)
DECLARE_GET_FROM_JSON(TGateImpl)
DECLARE_GET_FROM_JSON(TdagGateImpl)
DECLARE_GET_FROM_JSON(SqrtXGateImpl)
DECLARE_GET_FROM_JSON(SqrtXdagGateImpl)
DECLARE_GET_FROM_JSON(SqrtYGateImpl)
DECLARE_GET_FROM_JSON(SqrtYdagGateImpl)
DECLARE_GET_FROM_JSON(P0GateImpl)
DECLARE_GET_FROM_JSON(P1GateImpl)
#undef DECLARE_GET_FROM_JSON
#undef DECLARE_GET_FROM_JSON_WITH_TYPE

#define DECLARE_GET_FROM_JSON_RGATE_WITH_TYPE(Impl, Type)                   \
    template <>                                                             \
    inline std::shared_ptr<const Impl<Type>> get_from_json(const Json& j) { \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();    \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();  \
        Type angle = j.at("angle").get<Type>();                             \
        return std::make_shared<const Impl<Type>>(                          \
            vector_to_mask(targets), vector_to_mask(controls), angle);      \
    }
#define DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_TYPE(Type)    \
    DECLARE_GET_FROM_JSON_RGATE_WITH_TYPE(RXGateImpl, Type) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_TYPE(RYGateImpl, Type) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_TYPE(RZGateImpl, Type)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_RGATE_WITH_TYPE
#undef DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_TYPE

#define DECLARE_GET_FROM_JSON_UGATE_WITH_TYPE(Type)                                 \
    template <>                                                                     \
    inline std::shared_ptr<const U1GateImpl<Type>> get_from_json(const Json& j) {   \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();            \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();          \
        Type theta = j.at("theta").get<Type>();                                     \
        return std::make_shared<const U1GateImpl<Type>>(                            \
            vector_to_mask(targets), vector_to_mask(controls), theta);              \
    }                                                                               \
    template <>                                                                     \
    inline std::shared_ptr<const U2GateImpl<Type>> get_from_json(const Json& j) {   \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();            \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();          \
        Type theta = j.at("theta").get<Type>();                                     \
        Type phi = j.at("phi").get<Type>();                                         \
        return std::make_shared<const U2GateImpl<Type>>(                            \
            vector_to_mask(targets), vector_to_mask(controls), theta, phi);         \
    }                                                                               \
    template <>                                                                     \
    inline std::shared_ptr<const U3GateImpl<Type>> get_from_json(const Json& j) {   \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();            \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();          \
        Type theta = j.at("theta").get<Type>();                                     \
        Type phi = j.at("phi").get<Type>();                                         \
        Type lambda = j.at("lambda").get<Type>();                                   \
        return std::make_shared<const U3GateImpl<Type>>(                            \
            vector_to_mask(targets), vector_to_mask(controls), theta, phi, lambda); \
    }
DECLARE_GET_FROM_JSON_UGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_UGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_UGATE_WITH_TYPE

#define DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE(Type)                               \
    template <>                                                                      \
    inline std::shared_ptr<const SwapGateImpl<Type>> get_from_json(const Json& j) {  \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();             \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();           \
        return std::make_shared<const SwapGateImpl<Type>>(vector_to_mask(targets),   \
                                                          vector_to_mask(controls)); \
    }
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE*/

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <std::floating_point Fp, ExecutionSpace Sp>
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, Fp, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             Fp,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate<Fp, Sp>& gate) { return gate->phase(); },
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
            [](const U1Gate<Fp, Sp>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Fp,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate<Fp, Sp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Fp, Sp>& gate) { return gate->lambda(); },
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
            "theta",
            [](const U3Gate<Fp, Sp>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Fp, Sp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Fp, Sp>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(SwapGate, Fp, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
