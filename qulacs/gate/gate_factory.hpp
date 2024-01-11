#pragma once

#include "gate/gate.hpp"
#include "gate/gate_one_control_one_target.hpp"
#include "gate/gate_one_qubit.hpp"
#include "gate/gate_quantum_matrix.hpp"

namespace qulacs {
namespace gate {
GatePtr I(UINT target) { return std::make_unique<qulacs::I>(target); }
GatePtr X(UINT target) { return std::make_unique<qulacs::X>(target); }
GatePtr Y(UINT target) { return std::make_unique<qulacs::Y>(target); }
GatePtr Z(UINT target) { return std::make_unique<qulacs::Z>(target); }
GatePtr H(UINT target) { return std::make_unique<qulacs::H>(target); }
GatePtr S(UINT target) { return std::make_unique<qulacs::S>(target); }
GatePtr Sdag(UINT target) { return std::make_unique<qulacs::Sdag>(target); }
GatePtr T(UINT target) { return std::make_unique<qulacs::T>(target); }
GatePtr Tdag(UINT target) { return std::make_unique<qulacs::Tdag>(target); }
GatePtr sqrtX(UINT target) { return std::make_unique<qulacs::sqrtX>(target); }
GatePtr sqrtXdag(UINT target) { return std::make_unique<qulacs::sqrtXdag>(target); }
GatePtr sqrtY(UINT target) { return std::make_unique<qulacs::sqrtY>(target); }
GatePtr sqrtYdag(UINT target) { return std::make_unique<qulacs::sqrtYdag>(target); }
GatePtr P0(UINT target) { return std::make_unique<qulacs::P0>(target); }
GatePtr P1(UINT target) { return std::make_unique<qulacs::P1>(target); }
GatePtr RX(UINT target, double angle) { return std::make_unique<qulacs::RX>(target, angle); }
GatePtr RY(UINT target, double angle) { return std::make_unique<qulacs::RY>(target, angle); }
GatePtr RZ(UINT target, double angle) { return std::make_unique<qulacs::RZ>(target, angle); }
GatePtr U1(UINT target, double lambda) { return std::make_unique<qulacs::U1>(target, lambda); }
GatePtr U2(UINT target, double phi, double lambda) {
    return std::make_unique<qulacs::U2>(target, phi, lambda);
}
GatePtr U3(UINT target, double theta, double phi, double lambda) {
    return std::make_unique<qulacs::U3>(target, theta, phi, lambda);
}
GatePtr CNOT(UINT control, UINT target) { return std::make_unique<qulacs::CNOT>(control, target); }
GatePtr CZ(UINT control, UINT target) { return std::make_unique<qulacs::CZ>(control, target); }
}  // namespace gate
}  // namespace qulacs
