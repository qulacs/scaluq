#pragma once

#include "gate/gate_npair_qubit.hpp"
#include "gate/gate_one_control_one_target.hpp"
#include "gate/gate_one_qubit.hpp"
#include "gate/gate_pauli.hpp"
#include "gate/gate_quantum_matrix.hpp"
#include "gate/gate_two_qubit.hpp"
#include "gate/gate_zero_qubit.hpp"

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

inline Gate I() { return internal::GateFactory::create_gate<internal::IGateImpl>(); }
inline Gate GlobalPhase(double phase) {
    return internal::GateFactory::create_gate<internal::GlobalPhaseGateImpl>(phase);
}
inline Gate X(UINT target) {
    return internal::GateFactory::create_gate<internal::XGateImpl>(target);
}
inline Gate Y(UINT target) {
    return internal::GateFactory::create_gate<internal::YGateImpl>(target);
}
inline Gate Z(UINT target) {
    return internal::GateFactory::create_gate<internal::ZGateImpl>(target);
}
inline Gate H(UINT target) {
    return internal::GateFactory::create_gate<internal::HGateImpl>(target);
}
inline Gate S(UINT target) {
    return internal::GateFactory::create_gate<internal::SGateImpl>(target);
}
inline Gate Sdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SdagGateImpl>(target);
}
inline Gate T(UINT target) {
    return internal::GateFactory::create_gate<internal::TGateImpl>(target);
}
inline Gate Tdag(UINT target) {
    return internal::GateFactory::create_gate<internal::TdagGateImpl>(target);
}
inline Gate SqrtX(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtXGateImpl>(target);
}
inline Gate SqrtXdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtXdagGateImpl>(target);
}
inline Gate SqrtY(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtYGateImpl>(target);
}
inline Gate SqrtYdag(UINT target) {
    return internal::GateFactory::create_gate<internal::SqrtYdagGateImpl>(target);
}
inline Gate P0(UINT target) {
    return internal::GateFactory::create_gate<internal::P0GateImpl>(target);
}
inline Gate P1(UINT target) {
    return internal::GateFactory::create_gate<internal::P1GateImpl>(target);
}
inline Gate RX(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RXGateImpl>(target, angle);
}
inline Gate RY(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RYGateImpl>(target, angle);
}
inline Gate RZ(UINT target, double angle) {
    return internal::GateFactory::create_gate<internal::RZGateImpl>(target, angle);
}
inline Gate U1(UINT target, double lambda) {
    return internal::GateFactory::create_gate<internal::U1GateImpl>(target, lambda);
}
inline Gate U2(UINT target, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U2GateImpl>(target, phi, lambda);
}
inline Gate U3(UINT target, double theta, double phi, double lambda) {
    return internal::GateFactory::create_gate<internal::U3GateImpl>(target, theta, phi, lambda);
}
inline Gate CX(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CXGateImpl>(control, target);
}
auto& CNot = CX;
inline Gate CZ(UINT control, UINT target) {
    return internal::GateFactory::create_gate<internal::CZGateImpl>(control, target);
}
inline Gate Swap(UINT target1, UINT target2) {
    return internal::GateFactory::create_gate<internal::SwapGateImpl>(target1, target2);
}
inline Gate FusedSwap(UINT qubit_index1, UINT qubit_index2, UINT block_size) {
    return internal::GateFactory::create_gate<internal::FusedSwapGateImpl>(
        qubit_index1, qubit_index2, block_size);
}
inline Gate Pauli(const PauliOperator& pauli) {
    return internal::GateFactory::create_gate<internal::PauliGateImpl>(pauli);
}
inline Gate PauliRotation(const PauliOperator& pauli, double angle) {
    return internal::GateFactory::create_gate<internal::PauliRotationGateImpl>(pauli, angle);
}
}  // namespace qulacs
