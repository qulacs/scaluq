#pragma once

#include "gate/gate_npair_qubit.hpp"
#include "gate/gate_one_control_one_target.hpp"
#include "gate/gate_one_qubit.hpp"
#include "gate/gate_quantum_matrix.hpp"
#include "gate/gate_two_qubit.hpp"

namespace qulacs {
namespace internal {
class GateFactory {
public:
    template <GateImpl T, typename... Args>
    static Gate create_gate(Args... args) {
        return {std::make_shared<T>(args...)};
    }
};
}  // namespace internal

Gate I(UINT target) { return internal::GateFactory::create_gate<internal::IGateImpl>(target); }
Gate X(UINT target) { return internal::GateFactory::create_gate<internal::XGateImpl>(target); }
Gate Y(UINT target) { return internal::GateFactory::create_gate<internal::YGateImpl>(target); }
Gate Z(UINT target) { return internal::GateFactory::create_gate<internal::ZGateImpl>(target); }
Gate H(UINT target) { return internal::GateFactory::create_gate<internal::HGateImpl>(target); }
Gate S(UINT target) { return internal::GateFactory::create_gate<internal::SGateImpl>(target); }
Gate Sdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl>(target);
}
Gate T(UINT target) { return internal::GateFactory::create_gate<internal::TGateImpl>(target); }
Gate Tdag(UINT target) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl>(target);
}
Gate sqrtX(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtXGateImpl>(target);
}
Gate sqrtXdag(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtXdagGateImpl>(target);
}
Gate sqrtY(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtYGateImpl>(target);
}
Gate sqrtYdag(UINT target) {
    return internal::GateFactory::create_gate<internal::sqrtYdagGateImpl>(target);
}
Gate P0(UINT target) { return internal::GateFactory::create_gate<internal::P0GateImpl>(target); }
Gate P1(UINT target) { return internal::GateFactory::create_gate<internal::P1GateImpl>(target); }
Gate RX(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RXGateImpl>(target, angle);
}
Gate RY(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RYGateImpl>(target, angle);
}
Gate RZ(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RZGateImpl>(target, angle);
}
Gate U1(UINT target, double lambda) {
    return internal::GateFactory::create_gate<internal::U1GateImpl>(target, lambda);
}
Gate U2(UINT target, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U2GateImpl>(target, phi, lambda);
}
Gate U3(UINT target, double theta, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U3GateImpl>(target, theta, phi, lambda);
}
Gate CNOT(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CNOTGateImpl>(control, target);
}
Gate CZ(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CZGateImpl>(control, target);
}
Gate SWAP(UINT target1, UINT target2) {
    return internal::GateFactory::create_gate<internal::SWAPGateImpl>(target1, target2);
}
Gate FusedSWAP(UINT qubit_index1, UINT qubit_index2, UINT block_size) {
    return internal::GateFactory::create_gate<internal::FusedSWAPGateImpl>(
        qubit_index1, qubit_index2, block_size);
}
}  // namespace qulacs
