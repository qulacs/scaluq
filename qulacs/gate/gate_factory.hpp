#pragma once

#include "gate/gate.hpp"
#include "gate/gate_one_control_one_target.hpp"
#include "gate/gate_one_qubit.hpp"
#include "gate/gate_quantum_matrix.hpp"

namespace qulacs {
namespace internal {
class GateFactory {
public:
    template <GateImpl T, typename... Args>
    static Gate create_gate(Args... args) {
        return {std::make_unique<T>(args...)};
    }
};
}  // namespace internal

Gate I(UINT target) { return internal::GateFactory::create_gate<internal::IGate>(target); }
Gate X(UINT target) { return internal::GateFactory::create_gate<internal::XGate>(target); }
Gate Y(UINT target) { return internal::GateFactory::create_gate<internal::YGate>(target); }
Gate Z(UINT target) { return internal::GateFactory::create_gate<internal::ZGate>(target); }
Gate H(UINT target) { return internal::GateFactory::create_gate<internal::HGate>(target); }
Gate S(UINT target) { return internal::GateFactory::create_gate<internal::SGate>(target); }
Gate Sdag(UINT target) { return internal::GateFactory::create_gate<internal::SdagGate>(target); }
Gate T(UINT target) { return internal::GateFactory::create_gate<internal::TGate>(target); }
Gate Tdag(UINT target) { return internal::GateFactory::create_gate<internal::TdagGate>(target); }
Gate sqrtX(UINT target) { return internal::GateFactory::create_gate<internal::sqrtXGate>(target); }
Gate sqrtXdag(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtXdagGate>(target);
}
Gate sqrtY(UINT target) { return internal::GateFactory::create_gate<internal::sqrtYGate>(target); }
Gate sqrtYdag(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtYdagGate>(target);
}
Gate P0(UINT target) { return internal::GateFactory::create_gate<internal::P0Gate>(target); }
Gate P1(UINT target) { return internal::GateFactory::create_gate<internal::P1Gate>(target); }
Gate RX(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RXGate>(target, angle);
}
Gate RY(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RYGate>(target, angle);
}
Gate RZ(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RZGate>(target, angle);
}
Gate U1(UINT target, double lambda) {
    return internal::GateFactory::create_gate<internal::U1Gate>(target, lambda);
}
Gate U2(UINT target, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U2Gate>(target, phi, lambda);
}
Gate U3(UINT target, double theta, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U3Gate>(target, theta, phi, lambda);
}
Gate CNOT(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CNOTGate>(control, target);
}
Gate CZ(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CZGate>(control, target);
}
}  // namespace qulacs
