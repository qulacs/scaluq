#include "merge_gate.hpp"

#include "../constant.hpp"
#include "../util/utility.hpp"
#include "gate/gate_factory.hpp"

namespace scaluq {
/**
 * @details ignore global phase (because PauliRoation -> matrix ignore global phase)
 */
std::pair<Gate, double> merge_gate(const Gate& gate1, const Gate& gate2) {
    GateType gate_type1 = gate1.gate_type();
    GateType gate_type2 = gate2.gate_type();
    // TODO: Deal with ProbablisticGate

    // Special case: Zero qubit
    if (gate_type1 == GateType::I) return {gate2, 0.};  // copy can be removed by #125
    if (gate_type2 == GateType::I) return {gate1, 0.};
    if (gate_type1 == GateType::GlobalPhase) return {gate2, GlobalPhaseGate(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase) return {gate1, GlobalPhaseGate(gate2)->phase()};

    // Special case: Pauli
    auto get_pauli_id = [&](GateType gate_type) -> std::optional<std::uint64_t> {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::X) return 1;
        if (gate_type == GateType::Y) return 2;
        if (gate_type == GateType::Z) return 3;
        return std::nullopt;
    };
    auto pauli_id1 = get_pauli_id(gate_type1);
    auto pauli_id2 = get_pauli_id(gate_type2);
    assert(!pauli_id1 || pauli_id1 != 0);
    assert(!pauli_id2 || pauli_id2 != 0);
    if (pauli_id1 && pauli_id2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            if (pauli_id1 == pauli_id2) return {gate::I(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) return {gate::Z(target1), -Kokkos::numbers::pi / 2};
                if (pauli_id2 == 3) return {gate::Y(target1), Kokkos::numbers::pi / 2};
            }
            if (pauli_id1 == 2) {
                if (pauli_id2 == 3) return {gate::X(target1), -Kokkos::numbers::pi / 2};
                if (pauli_id2 == 1) return {gate::Z(target1), Kokkos::numbers::pi / 2};
            }
            if (pauli_id1 == 3) {
                if (pauli_id2 == 1) return {gate::Y(target1), -Kokkos::numbers::pi / 2};
                if (pauli_id2 == 2) return {gate::X(target1), Kokkos::numbers::pi / 2};
            }
        }
    }
    if ((pauli_id1 || gate1.gate_type() == GateType::Pauli) &&
        (pauli_id2 || gate2.gate_type() == GateType::Pauli)) {
        auto pauli1 = gate_type1 == GateType::Pauli
                          ? PauliGate(gate1)->pauli()
                          : PauliOperator(std::vector{gate1->target_qubit_list()[0]},
                                          std::vector{pauli_id1.value()});
        auto pauli2 = gate_type2 == GateType::Pauli
                          ? PauliGate(gate2)->pauli()
                          : PauliOperator(std::vector{gate2->target_qubit_list()[0]},
                                          std::vector{pauli_id2.value()});
        return {gate::Pauli(pauli2 * pauli1), 0.};
    }

    // Special case: Phase
    auto get_oct_phase = [&](GateType gate_type) -> std::optional<std::uint64_t> {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::Z) return 4;
        if (gate_type == GateType::S) return 2;
        if (gate_type == GateType::Sdag) return 6;
        if (gate_type == GateType::T) return 1;
        if (gate_type == GateType::Tdag) return 7;
        return std::nullopt;
    };
    auto oct_phase_gate = [&](std::uint64_t oct_phase,
                              std::uint64_t target) -> std::optional<Gate> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I();
        if (oct_phase == 4) return gate::Z(target);
        if (oct_phase == 2) return gate::S(target);
        if (oct_phase == 6) return gate::Sdag(target);
        if (oct_phase == 1) return gate::T(target);
        if (oct_phase == 7) return gate::Tdag(target);
        return std::nullopt;
    };
    auto oct_phase1 = get_oct_phase(gate_type1);
    auto oct_phase2 = get_oct_phase(gate_type2);
    if (oct_phase1 && oct_phase2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            auto g = oct_phase_gate(oct_phase1.value() + oct_phase2.value(), target1);
            if (g) return {g.value(), 0.};
        }
    }
    if ((oct_phase1 || gate_type1 == GateType::RZ || gate_type1 == GateType::U1) &&
        (oct_phase2 || gate_type2 == GateType::RZ || gate_type2 == GateType::U1)) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            double phase1 = oct_phase1 ? oct_phase1.value() * Kokkos::numbers::pi / 4
                            : gate_type1 == GateType::RZ ? RZGate(gate1)->angle()
                                                         : U1Gate(gate1)->lambda();
            double global_phase1 = gate_type1 == GateType::RZ ? -RZGate(gate1)->angle() / 2 : 0.;
            double phase2 = oct_phase2 ? oct_phase2.value() * Kokkos::numbers::pi / 4
                            : gate_type2 == GateType::RZ ? RZGate(gate2)->angle()
                                                         : U1Gate(gate2)->lambda();
            double global_phase2 = gate_type2 == GateType::RZ ? -RZGate(gate2)->angle() / 2 : 0.;
            return {gate::U1(target1, phase1 + phase2), global_phase1 + global_phase2};
        }
    }

    // Special case: RX
    auto get_rx_angle = [&](Gate gate, GateType gate_type) -> std::optional<double> {
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::X) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtX) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtXdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RX) return RXGate(gate)->angle();
        return std::nullopt;
    };
    auto rx_param1 = get_rx_angle(gate1, gate_type1);
    auto rx_param2 = get_rx_angle(gate2, gate_type2);
    if (rx_param1 && rx_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        double global_phase1 = gate_type1 == GateType::RX ? 0. : rx_param1.value() / 2;
        double global_phase2 = gate_type2 == GateType::RX ? 0. : rx_param2.value() / 2;
        if (target1 == target2) {
            return {gate::RX(target1, rx_param1.value() + rx_param2.value()),
                    global_phase1 + global_phase2};
        }
    }

    // Special case: RY
    auto get_ry_angle = [&](Gate gate, GateType gate_type) -> std::optional<double> {
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::Y) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtY) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtYdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RY) return RYGate(gate)->angle();
        return std::nullopt;
    };
    auto ry_param1 = get_ry_angle(gate1, gate_type1);
    auto ry_param2 = get_ry_angle(gate2, gate_type2);
    if (ry_param1 && ry_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        double global_phase1 = gate_type1 == GateType::RY ? 0. : ry_param1.value() / 2;
        double global_phase2 = gate_type2 == GateType::RY ? 0. : ry_param2.value() / 2;
        if (target1 == target2) {
            return {gate::RY(target1, ry_param1.value() + ry_param2.value()),
                    global_phase1 + global_phase2};
        }
    }

    // Special case: CX,CZ,Swap,FusedSwap duplication
    if (gate_type1 == gate_type2 && gate_type1 == GateType::CX) {
        CXGate cx1(gate1), cx2(gate2);
        if (cx1->target() == cx2->target() && cx1->control() == cx2->control())
            return {gate::I(), 0.};
    }
    if (gate_type1 == gate_type2 && gate_type1 == GateType::CZ) {
        CZGate cz1(gate1), cz2(gate2);
        if (cz1->target() == cz2->target() && cz1->control() == cz2->control())
            return {gate::I(), 0.};
        if (cz1->target() == cz2->control() && cz1->control() == cz2->target())
            return {gate::I(), 0.};
    }
    if (gate_type1 == gate_type2 && gate_type1 == GateType::Swap) {
        SwapGate swap1(gate1), swap2(gate2);
        if (swap1->target1() == swap2->target1() && swap1->target2() == swap2->target2())
            return {gate::I(), 0.};
        if (swap1->target1() == swap2->target2() && swap1->target2() == swap2->target1())
            return {gate::I(), 0.};
    }
    if (gate_type1 == gate_type2 && gate_type1 == GateType::FusedSwap) {
        FusedSwapGate swap1(gate1), swap2(gate2);
        if (swap1->block_size() == swap2->block_size()) {
            if (swap1->qubit_index1() == swap2->qubit_index1() &&
                swap1->qubit_index2() == swap2->qubit_index2())
                return {gate::I(), 0.};
            if (swap1->qubit_index1() == swap2->qubit_index2() &&
                swap1->qubit_index2() == swap2->qubit_index1())
                return {gate::I(), 0.};
        }
    }

    // General case
    auto gate1_targets = gate1->target_qubit_list();
    std::ranges::copy(gate1->control_qubit_list(), std::back_inserter(gate1_targets));
    auto gate2_targets = gate2->target_qubit_list();
    std::ranges::copy(gate2->control_qubit_list(), std::back_inserter(gate2_targets));
    std::vector<std::uint64_t> merged_targets(gate1_targets.size() + gate2_targets.size());
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
        internal::get_expanded_matrix(gate1->get_matrix().value(), gate1_targets, merged_targets);
    auto matrix2 =
        internal::get_expanded_matrix(gate2->get_matrix().value(), gate2_targets, merged_targets);
    auto matrix = matrix2 * matrix1;
    return {gate::DenseMatrix(merged_targets, matrix), 0.};
}
}  // namespace scaluq
