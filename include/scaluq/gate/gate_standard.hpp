#pragma once

#include "../constant.hpp"
#include "gate.hpp"

namespace scaluq {
namespace internal {

template <Precision Prec>
class IGateImpl : public GateBase<Prec> {
public:
    IGateImpl() : GateBase<Prec>(0, 0) {}

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override { j = Json{{"type", "I"}}; }
};

template <Precision Prec>
class GlobalPhaseGateImpl : public GateBase<Prec> {
protected:
    Float<Prec> _phase;

public:
    GlobalPhaseGateImpl(std::uint64_t control_mask, Float<Prec> phase)
        : GateBase<Prec>(0, control_mask), _phase(phase){};

    [[nodiscard]] double phase() const { return _phase; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const GlobalPhaseGateImpl<Prec>>(this->_control_mask, -_phase);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "GlobalPhase"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"phase", this->phase()}};
    }
};

template <Precision Prec>
class RotationGateBase : public GateBase<Prec> {
protected:
    Float<Prec> _angle;

public:
    RotationGateBase(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> angle)
        : GateBase<Prec>(target_mask, control_mask), _angle(angle) {}

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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "X"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Y"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Z"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "H"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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
        return std::make_shared<const SdagGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "S"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class SdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Sdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class TGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TdagGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "T"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class TdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const TGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Tdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class SqrtXGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXdagGateImpl<Prec>>(this->_target_mask,
                                                              this->_control_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class SqrtXdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtXGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtXdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class SqrtYGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYdagGateImpl<Prec>>(this->_target_mask,
                                                              this->_control_mask);
    }

    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class SqrtYdagGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const SqrtYGateImpl<Prec>>(this->_target_mask, this->_control_mask);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "SqrtYdag"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P0"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "P1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
    }
};

template <Precision Prec>
class RXGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RXGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RX"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class RYGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RYGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RY"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class RZGateImpl : public RotationGateBase<Prec> {
public:
    using RotationGateBase<Prec>::RotationGateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const RZGateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -this->_angle);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "RZ"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"angle", this->angle()}};
    }
};

template <Precision Prec>
class U1GateImpl : public GateBase<Prec> {
    Float<Prec> _lambda;

public:
    U1GateImpl(std::uint64_t target_mask, std::uint64_t control_mask, Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _lambda(lambda) {}

    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U1GateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -_lambda);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U1"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
                 {"lambda", this->lambda()}};
    }
};
template <Precision Prec>
class U2GateImpl : public GateBase<Prec> {
    Float<Prec> _phi, _lambda;

public:
    U2GateImpl(std::uint64_t target_mask,
               std::uint64_t control_mask,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _phi(phi), _lambda(lambda) {}

    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U2GateImpl<Prec>>(
            this->_target_mask,
            this->_control_mask,
            -_lambda - static_cast<Float<Prec>>(Kokkos::numbers::pi),
            -_phi + static_cast<Float<Prec>>(Kokkos::numbers::pi));
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "U2"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()},
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
               Float<Prec> theta,
               Float<Prec> phi,
               Float<Prec> lambda)
        : GateBase<Prec>(target_mask, control_mask), _theta(theta), _phi(phi), _lambda(lambda) {}

    double theta() const { return _theta; }
    double phi() const { return _phi; }
    double lambda() const { return _lambda; }

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return std::make_shared<const U3GateImpl<Prec>>(
            this->_target_mask, this->_control_mask, -_theta, -_lambda, -_phi);
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

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

template <Precision Prec>
class SwapGateImpl : public GateBase<Prec> {
public:
    using GateBase<Prec>::GateBase;

    std::shared_ptr<const GateBase<Prec>> get_inverse() const override {
        return this->shared_from_this();
    }
    ComplexMatrix get_matrix() const override;

    void update_quantum_state(StateVector<Prec>& state_vector) const override;
    void update_quantum_state(StateVectorBatched<Prec>& states) const override;

    std::string to_string(const std::string& indent) const override;

    void get_as_json(Json& j) const override {
        j = Json{{"type", "Swap"},
                 {"target", this->target_qubit_list()},
                 {"control", this->control_qubit_list()}};
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

namespace internal {

#define DECLARE_GET_FROM_JSON_IGATE_WITH_PRECISION(Prec)                       \
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

}  // namespace internal

#ifdef SCALUQ_USE_NANOBIND
namespace internal {
template <Precision Prec>
void bind_gate_gate_standard_hpp(nb::module_& m) {
    DEF_GATE(IGate, Prec, "Specific class of Pauli-I gate.");
    DEF_GATE(GlobalPhaseGate,
             Prec,
             "Specific class of gate, which rotate global phase, represented as "
             "$e^{i\\mathrm{phase}}I$.")
        .def(
            "phase",
            [](const GlobalPhaseGate<Prec>& gate) { return gate->phase(); },
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
            [](const U1Gate<Prec>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(U2Gate,
             Prec,
             "Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, "
             "represented as "
             "$\\frac{1}{\\sqrt{2}} \\begin{bmatrix}1 & -e^{-i\\lambda}\\\\\n"
             "e^{i\\phi} & e^{i(\\phi+\\lambda)}\n\\end{bmatrix}$.")
        .def(
            "phi", [](const U2Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U2Gate<Prec>& gate) { return gate->lambda(); },
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
            [](const U3Gate<Prec>& gate) { return gate->theta(); },
            "Get `theta` property.")
        .def(
            "phi", [](const U3Gate<Prec>& gate) { return gate->phi(); }, "Get `phi` property.")
        .def(
            "lambda_",
            [](const U3Gate<Prec>& gate) { return gate->lambda(); },
            "Get `lambda` property.");
    DEF_GATE(SwapGate, Prec, "Specific class of two-qubit swap gate.");
}
}  // namespace internal
#endif
}  // namespace scaluq
