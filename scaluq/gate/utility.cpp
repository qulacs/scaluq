#include "utility.hpp"

#include <set>

#include "../constant.hpp"
#include "../util/utility.hpp"
#include "gate/gate_factory.hpp"

namespace scaluq {
std::pair<Gate, double> merge_gate(const Gate& gate1, const Gate& gate2) {
    GateType gate_type1 = gate1.gate_type();
    GateType gate_type2 = gate2.gate_type();
    // TODO: Deal with ProbablisticGate

    // Special case: Zero qubit
    if (gate_type1 == GateType::I) return {gate2->copy(), 0.};  // copy can be removed by #125
    if (gate_type2 == GateType::I) return {gate1->copy(), 0.};
    if (gate_type1 == GateType::GlobalPhase)
        return {gate2->copy(), GlobalPhaseGate(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase)
        return {gate1->copy(), GlobalPhaseGate(gate2)->phase()};

    // Special case: Pauli
    constexpr UINT NOT_SINGLE_PAULI = std::numeric_limits<UINT>::max();
    auto get_pauli_id = [&](GateType gate_type) -> UINT {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::X) return 1;
        if (gate_type == GateType::Y) return 2;
        if (gate_type == GateType::Z) return 3;
        return NOT_SINGLE_PAULI;
    };
    UINT pauli_id1 = get_pauli_id(gate_type1);
    UINT pauli_id2 = get_pauli_id(gate_type2);
    assert(pauli_id1 != 0);
    assert(pauli_id2 != 0);
    if (pauli_id1 != NOT_SINGLE_PAULI && pauli_id2 != NOT_SINGLE_PAULI) {
        UINT target1 = gate1->get_target_qubit_list()[0];
        UINT target2 = gate2->get_target_qubit_list()[0];
        if (target1 == target2) {
            if (pauli_id1 == pauli_id2) return {gate::I(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) return {gate::Z(target1), -PI() / 2};
                if (pauli_id2 == 3) return {gate::Y(target1), PI() / 2};
            }
            if (pauli_id1 == 2) {
                if (pauli_id2 == 3) return {gate::X(target1), -PI() / 2};
                if (pauli_id2 == 1) return {gate::Z(target1), PI() / 2};
            }
            if (pauli_id1 == 3) {
                if (pauli_id2 == 1) return {gate::Y(target1), -PI() / 2};
                if (pauli_id2 == 2) return {gate::X(target1), PI() / 2};
            }
        }
    }
    if ((pauli_id1 != NOT_SINGLE_PAULI || gate1.gate_type() == GateType::Pauli) &&
        (pauli_id2 != NOT_SINGLE_PAULI || gate2.gate_type() == GateType::Pauli)) {
        auto pauli1 = gate_type1 == GateType::Pauli
                          ? PauliGate(gate1)->pauli()
                          : PauliOperator(std::vector{gate1->get_target_qubit_list()[0]},
                                          std::vector{pauli_id1});
        auto pauli2 = gate_type2 == GateType::Pauli
                          ? PauliGate(gate2)->pauli()
                          : PauliOperator(std::vector{gate2->get_target_qubit_list()[0]},
                                          std::vector{pauli_id2});
        return {gate::Pauli(pauli1 * pauli2), 0.};
    }

    // Special case: Phase
    constexpr UINT NOT_FIXED_PHASE = std::numeric_limits<UINT>::max();
    auto get_oct_phase = [&](GateType gate_type) -> UINT {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::Z) return 4;
        if (gate_type == GateType::S) return 2;
        if (gate_type == GateType::Sdag) return 6;
        if (gate_type == GateType::T) return 1;
        if (gate_type == GateType::Tdag) return 7;
        return NOT_FIXED_PHASE;
    };
    auto oct_phase_gate = [&](UINT oct_phase, UINT target) -> std::optional<Gate> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I();
        if (oct_phase == 4) return gate::Z(target);
        if (oct_phase == 2) return gate::S(target);
        if (oct_phase == 6) return gate::Sdag(target);
        if (oct_phase == 1) return gate::T(target);
        if (oct_phase == 7) return gate::Tdag(target);
        return std::nullopt;
    };
    UINT oct_phase1 = get_oct_phase(gate_type1);
    UINT oct_phase2 = get_oct_phase(gate_type2);
    if (oct_phase1 != NOT_FIXED_PHASE && oct_phase2 != NOT_FIXED_PHASE) {
        UINT target1 = gate1->get_target_qubit_list()[0];
        UINT target2 = gate2->get_target_qubit_list()[0];
        if (target1 == target2) {
            auto g = oct_phase_gate(oct_phase1 + oct_phase2, target1);
            if (g) return {g.value(), 0.};
        }
    }
    if ((oct_phase1 != NOT_FIXED_PHASE || gate_type1 == GateType::RZ ||
         gate_type1 == GateType::U1) &&
        (oct_phase2 != NOT_FIXED_PHASE || gate_type2 == GateType::RZ ||
         gate_type2 == GateType::U1)) {
        UINT target1 = gate1->get_target_qubit_list()[0];
        UINT target2 = gate2->get_target_qubit_list()[0];
        if (target1 == target2) {
            double phase1 = oct_phase1 != NOT_FIXED_PHASE ? oct_phase1 * PI() / 4
                            : gate_type1 == GateType::RZ  ? RZGate(gate1)->angle()
                                                          : U1Gate(gate1)->lambda();
            double global_phase1 = gate_type1 == GateType::RZ ? -RZGate(gate1)->angle() / 2 : 0.;
            double phase2 = oct_phase2 != NOT_FIXED_PHASE ? oct_phase2 * PI() / 4
                            : gate_type2 == GateType::RZ  ? RZGate(gate2)->angle()
                                                          : U1Gate(gate2)->lambda();
            double global_phase2 = gate_type2 == GateType::RZ ? -RZGate(gate2)->angle() / 2 : 0.;
            return {gate::U1(target1, phase1 + phase2), global_phase1 + global_phase2};
        }
    }

    // Special case: RX
    constexpr double NOT_RX = std::numeric_limits<double>::quiet_NaN();
    auto get_rx_angle = [&](Gate gate, GateType gate_type) -> std::pair<double, double> {
        if (gate_type == GateType::I) return {0., 0.};
        if (gate_type == GateType::X) return {PI(), PI() / 2};
        if (gate_type == GateType::SqrtX) return {PI() / 2, PI() / 4};
        if (gate_type == GateType::SqrtXdag) return {-PI() / 2, -PI() / 4};
        if (gate_type == GateType::RX) return {RXGate(gate)->angle(), 0.};
        return {NOT_RX, NOT_RX};
    };
    auto rx_param1 = get_rx_angle(gate1, gate_type1);
    auto rx_param2 = get_rx_angle(gate2, gate_type2);
    if (rx_param1.first != NOT_RX && rx_param2.first != NOT_RX) {
        UINT target1 = gate1->get_target_qubit_list()[0];
        UINT target2 = gate2->get_target_qubit_list()[0];
        if (target1 == target2) {
            return {gate::RX(target1, rx_param1.first + rx_param2.first),
                    rx_param1.second + rx_param2.second};
        }
    }

    // Special case: RY
    constexpr double NOT_RY = std::numeric_limits<double>::quiet_NaN();
    auto get_ry_angle = [&](Gate gate, GateType gate_type) -> std::pair<double, double> {
        if (gate_type == GateType::I) return {0., 0.};
        if (gate_type == GateType::Y) return {PI(), PI() / 2};
        if (gate_type == GateType::SqrtY) return {PI() / 2, PI() / 4};
        if (gate_type == GateType::SqrtYdag) return {-PI() / 2, -PI() / 4};
        if (gate_type == GateType::RY) return {RYGate(gate)->angle(), 0.};
        return {NOT_RY, NOT_RY};
    };
    auto ry_param1 = get_ry_angle(gate1, gate_type1);
    auto ry_param2 = get_ry_angle(gate2, gate_type2);
    if (ry_param1.first != NOT_RY && ry_param2.first != NOT_RY) {
        UINT target1 = gate1->get_target_qubit_list()[0];
        UINT target2 = gate2->get_target_qubit_list()[0];
        if (target1 == target2) {
            return {gate::RY(target1, ry_param1.first + ry_param2.first),
                    ry_param1.second + ry_param2.second};
        }
    }

    auto gate1_targets = gate1->get_target_qubit_list();
    std::ranges::copy(gate1->get_control_qubit_list(), std::back_inserter(gate1_targets));
    auto gate2_targets = gate2->get_target_qubit_list();
    std::ranges::copy(gate2->get_control_qubit_list(), std::back_inserter(gate2_targets));
    std::vector<UINT> merged_targets(gate1_targets.size() + gate2_targets.size());
    std::ranges::copy(gate1_targets, merged_targets.begin());
    std::ranges::copy(gate2_targets, merged_targets.begin() + gate1_targets.size());
    std::ranges::sort(merged_targets);
    merged_targets.erase(std::ranges::unique(merged_targets).begin(), merged_targets.end());
    if (merged_targets.size() >= 3) {
        throw std::runtime_error(
            "gate::merge_gate: Result gate's target size is equal or more than three. This is "
            "currently not implemented.");
    }
    auto matrix1 =
        internal::get_expaded_matrix(gate1->get_matrix().value(), gate1_targets, merged_targets);
    auto matrix2 =
        internal::get_expaded_matrix(gate2->get_matrix().value(), gate2_targets, merged_targets);
    auto matrix = matrix2 * matrix1;
    return {gate::DenseMatrix(merged_targets, matrix), 0.};
}
}  // namespace scaluq
