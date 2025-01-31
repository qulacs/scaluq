#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

<<<<<<< HEAD
template <Precision Prec>
class IGateImpl : public GateBase<Prec> {
public:
    IGateImpl() : GateBase<Prec>(0, 0) {}

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class IGateImpl : public GateBase<Fp, Sp> {
public:
    IGateImpl() : GateBase<Fp, Sp>(0, 0) {}

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override { j = Json{{"type", "I"}}; }
};

<<<<<<< HEAD
template <Precision Prec>
class GlobalPhaseGateImpl : public GateBase<Prec> {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class GlobalPhaseGateImpl : public GateBase<Fp, Sp> {
>>>>>>> set-space
protected:
    Float<Prec> _phase;

public:
<<<<<<< HEAD
    GlobalPhaseGateImpl(std::uint64_t control_mask, Float<Prec> phase)
        : GateBase<Prec>(0, control_mask), _phase(phase){};
=======
    GlobalPhaseGateImpl(std::uint64_t control_mask, Fp phase)
        : GateBase<Fp, Sp>(0, control_mask), _phase(phase){};
>>>>>>> set-space

    [[nodiscard]] double phase() const { return _phase; }

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Prec>>(this->_control_mask, -_phase);
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Fp, Sp>>(this->_control_mask, -_phase);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"phase", this->phase()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class RotationGateBase : public GateBase<Prec> {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class RotationGateBase : public GateBase<Fp, Sp> {
>>>>>>> set-space
protected:
    Float<Prec> _angle;

public:
<<<<<<< HEAD
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> angle)
        : GateBase<Prec>(target_mask, control_mask), _angle(angle) {}
=======
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Fp angle)
        : GateBase<Fp, Sp>(target_mask, control_mask), _angle(angle) {}
>>>>>>> set-space

    double angle() const { return _angle; }
};

<<<<<<< HEAD
template <Precision Prec>
class XGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class XGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "X"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class YGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class YGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Y"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class ZGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class ZGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Z"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class HGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class HGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "H"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
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
        return std::make_shared<const SdagGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
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
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Fp, Sp>>(this->_target_mask, this->_control_mask);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class TGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class TGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                            this->_control_mask);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class TdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class TdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Fp, Sp>>(this->_target_mask, this->_control_mask);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SqrtXGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Prec>>(this->_target_mask,
                                                              this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                                this->_control_mask);
>>>>>>> set-space
    }

    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SqrtXdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtXdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Fp, Sp>>(this->_target_mask,
                                                             this->_control_mask);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SqrtYGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Prec>>(this->_target_mask,
                                                              this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Fp, Sp>>(this->_target_mask,
                                                                this->_control_mask);
>>>>>>> set-space
    }

    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class SqrtYdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Prec>>(this->_target_mask, this->_control_mask);
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SqrtYdagGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Fp, Sp>>(this->_target_mask,
                                                             this->_control_mask);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class P0GateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class P0GateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        throw std::runtime_error("P0::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P0"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class P1GateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class P1GateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        throw std::runtime_error("P1::get_inverse: Projection gate doesn't have inverse gate");
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class RXGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class RXGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class RYGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class RYGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class RZGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Prec>>(
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class RZGateImpl : public RotationGateBase<Fp, Sp> {
public:
    using RotationGateBase<Fp, Sp>::RotationGateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class U1GateImpl : public GateBase<Prec> {
    Float<Prec> _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _lambda(lambda) {}
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class U1GateImpl : public GateBase<Fp, Sp> {
    Fp _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _lambda(lambda) {}
>>>>>>> set-space

    double lambda() const { return _lambda; }

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Prec>>(
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -_lambda);
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()}};
    }
};
<<<<<<< HEAD
template <Precision Prec>
class U2GateImpl : public GateBase<Prec> {
    Float<Prec> _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class U2GateImpl : public GateBase<Fp, Sp> {
    Fp _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp phi, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}
>>>>>>> set-space

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Prec>>(
            this->_target_mask,
            this->_control_mask,
            -_lambda - static_cast<Float<Prec>>(Kokkos::numbers::pi),
            -_phi + static_cast<Float<Prec>>(Kokkos::numbers::pi));
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Fp, Sp>>(this->_target_mask,
                                                          this->_control_mask,
                                                          -_lambda - Kokkos::numbers::pi,
                                                          -_phi + Kokkos::numbers::pi);
>>>>>>> set-space
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U2"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()},
                 {"phi", this->phi()}};
    }
};

<<<<<<< HEAD
template <Precision Prec>
class U3GateImpl : public GateBase<Prec> {
    Float<Prec> _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               Float<Prec> theta,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class U3GateImpl : public GateBase<Fp, Sp> {
    Fp _theta, _phi, _lambda;

public:
    U3GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Fp theta, Fp phi, Fp lambda)
        : GateBase<Fp, Sp>(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}
>>>>>>> set-space

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

<<<<<<< HEAD
    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Prec>>(
=======
    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Fp, Sp>>(
>>>>>>> set-space
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

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

<<<<<<< HEAD
template <Precision Prec>
class SwapGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
=======
template <std::floating_point Fp, ExecutionSpace Sp>
class SwapGateImpl : public GateBase<Fp, Sp> {
public:
    using GateBase<Fp, Sp>::GateBase;

    std::shared_ptr<const GateBase<Fp, Sp>> get_inverse() const override {
>>>>>>> set-space
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

<<<<<<< HEAD
    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;
=======
    void update_quantum_state(StateVector<Fp, Sp>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Fp, Sp>& states) const override;
>>>>>>> set-space

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Swap"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

}  // namespace internal

<<<<<<< HEAD
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

namespace internal {

#define DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Prec)                       \
=======
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
>>>>>>> set-space
    template <>                                                                \
    inline std::shared_ptr<const IGateImpl<Prec>> get_from_json(const Json&) { \
        return std::make_shared<const IGateImpl<Prec>>();                      \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION

#define DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION(Prec)                                 \
    template <>                                                                                    \
    inline std::shared_ptr<const GlobalPhaseGateImpl<Prec>> get_from_json(const Json& j) {         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                         \
        double phase = j.at("phase").get<double>();                                                \
        return std::make_shared<const GlobalPhaseGateImpl<Prec>>(vector_to_mask(controls),         \
                                                                 static_cast<Float<Prec>>(phase)); \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_GLOBALPHASEGATE_WITH_PRECISION

#define DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(Impl, Prec)    \
    template <>                                                              \
    inline std::shared_ptr<const Impl<Prec>> get_from_json(const Json& j) {  \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();     \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();   \
        return std::make_shared<const Impl<Prec>>(vector_to_mask(targets),   \
                                                  vector_to_mask(controls)); \
    }
#define DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION(Prec)          \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(XGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(YGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(ZGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(HGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SdagGateImpl, Prec)     \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(TGateImpl, Prec)        \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(TdagGateImpl, Prec)     \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SqrtXGateImpl, Prec)    \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SqrtXdagGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SqrtYGateImpl, Prec)    \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(SqrtYdagGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(P0GateImpl, Prec)       \
    DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION(P1GateImpl, Prec)
#ifdef SCALUQ_FLOAT16
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECALRE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_SINGLETARGETGATE_WITH_PRECISION
#undef DECLARE_GET_FROM_JSON_EACH_SINGLETARGETGATE_WITH_PRECISION

#define DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION(Impl, Prec)                                   \
    template <>                                                                                  \
    inline std::shared_ptr<const Impl<Prec>> get_from_json(const Json& j) {                      \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                       \
        double angle = j.at("angle").get<double>();                                              \
        return std::make_shared<const Impl<Prec>>(                                               \
            vector_to_mask(targets), vector_to_mask(controls), static_cast<Float<Prec>>(angle)); \
    }
#define DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION(Prec)    \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION(RXGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION(RYGateImpl, Prec) \
    DECLARE_GET_FROM_JSON_RGATE_WITH_PRECISION(RZGateImpl, Prec)
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_RGATE
#undef DECLARE_GET_FROM_JSON_EACH_RGATE_WITH_PRECISION

#define DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION(Prec)                                         \
    template <>                                                                                  \
    inline std::shared_ptr<const U1GateImpl<Prec>> get_from_json(const Json& j) {                \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                       \
        double theta = j.at("theta").get<double>();                                              \
        return std::make_shared<const U1GateImpl<Prec>>(                                         \
            vector_to_mask(targets), vector_to_mask(controls), static_cast<Float<Prec>>(theta)); \
    }                                                                                            \
    template <>                                                                                  \
    inline std::shared_ptr<const U2GateImpl<Prec>> get_from_json(const Json& j) {                \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                       \
        double theta = j.at("theta").get<double>();                                              \
        double phi = j.at("phi").get<double>();                                                  \
        return std::make_shared<const U2GateImpl<Prec>>(vector_to_mask(targets),                 \
                                                        vector_to_mask(controls),                \
                                                        static_cast<Float<Prec>>(theta),         \
                                                        static_cast<Float<Prec>>(phi));          \
    }                                                                                            \
    template <>                                                                                  \
    inline std::shared_ptr<const U3GateImpl<Prec>> get_from_json(const Json& j) {                \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();                         \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();                       \
        double theta = j.at("theta").get<double>();                                              \
        double phi = j.at("phi").get<double>();                                                  \
        double lambda = j.at("lambda").get<double>();                                            \
        return std::make_shared<const U3GateImpl<Prec>>(vector_to_mask(targets),                 \
                                                        vector_to_mask(controls),                \
                                                        static_cast<Float<Prec>>(theta),         \
                                                        static_cast<Float<Prec>>(phi),           \
                                                        static_cast<Float<Prec>>(lambda));       \
    }
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_UGATE_WITH_PRECISION

#define DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION(Prec)                          \
    template <>                                                                      \
    inline std::shared_ptr<const SwapGateImpl<Prec>> get_from_json(const Json& j) {  \
        auto targets = j.at("target").get<std::vector<std::uint64_t>>();             \
        auto controls = j.at("control").get<std::vector<std::uint64_t>>();           \
        return std::make_shared<const SwapGateImpl<Prec>>(vector_to_mask(targets),   \
                                                          vector_to_mask(controls)); \
    }
<<<<<<< HEAD
#ifdef SCALUQ_FLOAT16
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION(Precision::F16)
#endif
#ifdef SCALUQ_FLOAT32
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION(Precision::F32)
#endif
#ifdef SCALUQ_FLOAT64
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION(Precision::F64)
#endif
#ifdef SCALUQ_BFLOAT16
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION(Precision::BF16)
#endif
#undef DECLARE_GET_FROM_JSON_SWAPGATE_WITH_PRECISION
=======
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE(double)
DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE(float)
#undef DECLARE_GET_FROM_JSON_SWAPGATE_WITH_TYPE*/
>>>>>>> set-space

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
<<<<<<< HEAD
template <Precision Prec>
=======
template <std::floating_point Fp, ExecutionSpace Sp>
>>>>>>> set-space
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, Prec, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             Prec,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
<<<<<<< HEAD
            [](const GlobalPhaseGate<Prec>& gate) { return gate->phase(); },
=======
            [](const GlobalPhaseGate<Fp, Sp>& gate) { return gate->phase(); },
>>>>>>> set-space
            "Get `phase` property");
    DEF_GATE(XGate, Prec, "Specific class of Pauli-X gate.");
    DEF_GATE(YGate, Prec, "Specific class of Pauli-Y gate.");
    DEF_GATE(ZGate, Prec, "Specific class of Pauli-Z gate.");
    DEF_GATE(HGate, Prec, "Specific class of Hadamard gate.");
    DEF_GATE(SGate,
             Prec,
             "Specific class of S gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "i\n\\end{bmatrix}$.");
    DEF_GATE(SdagGate, Prec, "Specific class of inverse of S gate.");
    DEF_GATE(TGate,
             Prec,
             "Specific class of T gate, represented as $\\begin { bmatrix }\n1 & 0\\\\\n0 &"
             "e^{i\\pi/4}\n\\end{bmatrix}$.");
    DEF_GATE(TdagGate, Prec, "Specific class of inverse of T gate.");
    DEF_GATE(
        SqrtXGate,
        Prec,
        "Specific class of sqrt(X) gate, represented as $\\begin{ bmatrix }\n1+i & 1-i\\\\\n1-i "
        "& 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtXdagGate, Prec, "Specific class of inverse of sqrt(X) gate.");
    DEF_GATE(SqrtYGate,
             Prec,
             "Specific class of sqrt(Y) gate, represented as $\\begin{ bmatrix }\n1+i & -1-i "
             "\\\\\n1+i & 1+i\n\\end{bmatrix}$.");
    DEF_GATE(SqrtYdagGate, Prec, "Specific class of inverse of sqrt(Y) gate.");
    DEF_GATE(
        P0Gate,
        Prec,
        "Specific class of projection gate to $\\ket{0}$.\n\n.. note:: This gate is not unitary.");
    DEF_GATE(
        P1Gate,
        Prec,
        "Specific class of projection gate to $\\ket{1}$.\n\n.. note:: This gate is not unitary.");

#define DEF_ROTATION_GATE(GATE_TYPE, PRECISION, DESCRIPTION)                \
    DEF_GATE(GATE_TYPE, PRECISION, DESCRIPTION)                             \
        .def(                                                               \
            "angle",                                                        \
            [](const GATE_TYPE<PRECISION>& gate) { return gate->angle(); }, \
            "Get `angle` property.")

    DEF_ROTATION_GATE(
        RXGate,
        Prec,
        "Specific class of X rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}X}$.");
    DEF_ROTATION_GATE(
        RYGate,
        Prec,
        "Specific class of Y rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Y}$.");
    DEF_ROTATION_GATE(
        RZGate,
        Prec,
        "Specific class of Z rotation gate, represented as $e^{-i\\frac{\\mathrm{angle}}{2}Z}$.");

    DEF_GATE(U1Gate,
             Prec,
             "Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, "
             "represented as "
             "$\\begin{bmatrix}\n1 & 0\\\\\n0 & e^{i\\lambda}\n\\end{bmatrix}$.")
        .def(
            "lambda_",
<<<<<<< HEAD
            [](const U1Gate<Prec>& gate) { return gate->lambda(); },
=======
            [](const U1Gate<Fp, Sp>& gate) { return gate->lambda(); },
>>>>>>> set-space
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Prec,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
        .def(
<<<<<<< HEAD
            "phi", [](const U2Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Prec>& gate) { return gate->lambda(); },
=======
            "phi", [](const U2Gate<Fp, Sp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Fp, Sp>& gate) { return gate->lambda(); },
>>>>>>> set-space
            "Get `lambda` property.");
    DEF_GATE(U3Gate,
             Prec,
             "Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, "
             "represented as "
             "$\\begin{bmatrix}\n\\cos \\frac{\\theta}{2} & "
             "-e^{i\\lambda}\\sin\\frac{\\theta}{2}\\\\\n"
             "e^{i\\phi}\\sin\\frac{\\theta}{2} & "
             "e^{i(\\phi+\\lambda)}\\cos\\frac{\\theta}{2}\n\\end{bmatrix}$.")
        .def(
            "theta",
<<<<<<< HEAD
            [](const U3Gate<Prec>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Prec>& gate) { return gate->lambda(); },
=======
            [](const U3Gate<Fp, Sp>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Fp, Sp>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Fp, Sp>& gate) { return gate->lambda(); },
>>>>>>> set-space
            "Get `lambda` property.");
    DEF_GATE(SwapGate, Prec, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
