#pragma once

#include "gate/gate.hpp"
#include "gate/gate_one_control_one_target.hpp"
#include "gate/gate_one_qubit.hpp"

namespace qulacs {
namespace gate {
Gate I(UINT target) { return std::make_unique<qulacs::I>(target); }
Gate X(UINT target) { return std::make_unique<qulacs::X>(target); }
Gate Y(UINT target) { return std::make_unique<qulacs::Y>(target); }
Gate Z(UINT target) { return std::make_unique<qulacs::Z>(target); }
Gate H(UINT target) { return std::make_unique<qulacs::H>(target); }
Gate S(UINT target) { return std::make_unique<qulacs::S>(target); }
Gate Sdag(UINT target) { return std::make_unique<qulacs::Sdag>(target); }
Gate T(UINT target) { return std::make_unique<qulacs::T>(target); }
Gate Tdag(UINT target) { return std::make_unique<qulacs::Tdag>(target); }
Gate sqrtX(UINT target) { return std::make_unique<qulacs::sqrtX>(target); }
Gate sqrtXdag(UINT target) { return std::make_unique<qulacs::sqrtXdag>(target); }
Gate sqrtY(UINT target) { return std::make_unique<qulacs::sqrtY>(target); }
Gate sqrtYdag(UINT target) { return std::make_unique<qulacs::sqrtYdag>(target); }
Gate P0(UINT target) { return std::make_unique<qulacs::P0>(target); }
Gate P1(UINT target) { return std::make_unique<qulacs::P1>(target); }
Gate RX(UINT target, double angle) { return std::make_unique<qulacs::RX>(target, angle); }
Gate RY(UINT target, double angle) { return std::make_unique<qulacs::RY>(target, angle); }
Gate RZ(UINT target, double angle) { return std::make_unique<qulacs::RZ>(target, angle); }
Gate CNOT(UINT control, UINT target) { return std::make_unique<qulacs::CNOT>(control, target); }
Gate CZ(UINT control, UINT target) { return std::make_unique<qulacs::CZ>(control, target); }
}  // namespace gate
}  // namespace qulacs
