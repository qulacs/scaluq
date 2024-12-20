#include <scaluq/constant.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/util/utility.hpp>

#include "../util/template.hpp"

namespace scaluq {
FLOAT(Fp)
std::pair<Gate<Fp>, Fp> merge_gate_dense_matrix(const Gate<Fp>& gate1, const Gate<Fp>& gate2) {
    auto common_control_mask = gate1->control_qubit_mask() & gate2->control_qubit_mask();
    auto merged_operand_mask =
        (gate1->operand_qubit_mask() | gate2->operand_qubit_mask()) & ~common_control_mask;
    auto merged_operand_vector = internal::mask_to_vector(merged_operand_mask);
    auto matrix1 = internal::get_expanded_matrix(gate1->get_matrix(),
                                                 gate1->target_qubit_list(),
                                                 gate1->control_qubit_mask() & ~common_control_mask,
                                                 merged_operand_vector);
    auto matrix2 = internal::get_expanded_matrix(gate2->get_matrix(),
                                                 gate2->target_qubit_list(),
                                                 gate2->control_qubit_mask() & ~common_control_mask,
                                                 merged_operand_vector);
    std::cerr << matrix1 << std::endl;
    std::cerr << matrix2 << std::endl;
    auto matrix = matrix2 * matrix1;
    std::cerr << matrix << std::endl;
    return {gate::DenseMatrix<Fp>(
                merged_operand_vector, matrix, internal::mask_to_vector(common_control_mask)),
            0.};
}

FLOAT(Fp)
std::pair<Gate<Fp>, Fp> merge_gate(const Gate<Fp>& gate1, const Gate<Fp>& gate2) {
    GateType gate_type1 = gate1.gate_type();
    GateType gate_type2 = gate2.gate_type();

    if (gate_type1 == GateType::Probablistic || gate_type2 == GateType::Probablistic) {
        throw std::runtime_error(
            "merge_gate(const Gate<Fp>&, const Gate<Fp>&): ProbablisticGate is not supported.");
    }

    if (gate_type1 == GateType::I) return {gate2, 0.};
    if (gate_type2 == GateType::I) return {gate1, 0.};

    auto gate1_control_mask = gate1->control_qubit_mask();
    auto gate2_control_mask = gate2->control_qubit_mask();

    if (gate_type1 == GateType::GlobalPhase && gate1_control_mask == 0)
        return {gate2, GlobalPhaseGate<Fp>(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase && gate2_control_mask == 0)
        return {gate1, GlobalPhaseGate<Fp>(gate2)->phase()};

    if (gate1_control_mask != gate2_control_mask) return merge_gate_dense_matrix(gate1, gate2);
    auto control_list = internal::mask_to_vector(gate1_control_mask);

    // Special case: Zero qubit
    if (gate_type1 == GateType::GlobalPhase && gate_type2 == GateType::GlobalPhase) {
        return {gate::GlobalPhase<Fp>(
                    GlobalPhaseGate<Fp>(gate1)->phase() + GlobalPhaseGate<Fp>(gate2)->phase(),
                    control_list),
                0.};
    }

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
            if (pauli_id1 == pauli_id2) return {gate::I<Fp>(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<Fp>(target1, control_list), -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
                        return {gate::Y<Fp>(target1, control_list), Kokkos::numbers::pi / 2};
                    }
                }
            }
            if (pauli_id1 == 2) {
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
                        return {gate::X<Fp>(target1, control_list), -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<Fp>(target1, control_list), Kokkos::numbers::pi / 2};
                    }
                }
            }
            if (pauli_id1 == 3) {
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
                        return {gate::Y<Fp>(target1, control_list), -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::X<Fp>(target1, control_list), Kokkos::numbers::pi / 2};
                    }
                }
            }
        }
    }
    if ((pauli_id1 || gate1.gate_type() == GateType::Pauli) &&
        (pauli_id2 || gate2.gate_type() == GateType::Pauli)) {
        auto pauli1 = gate_type1 == GateType::Pauli
                          ? PauliGate<Fp>(gate1)->pauli()
                          : PauliOperator<Fp>(std::vector{gate1->target_qubit_list()[0]},
                                              std::vector{pauli_id1.value()});
        auto pauli2 = gate_type2 == GateType::Pauli
                          ? PauliGate<Fp>(gate2)->pauli()
                          : PauliOperator<Fp>(std::vector{gate2->target_qubit_list()[0]},
                                              std::vector{pauli_id2.value()});
        return {gate::Pauli<Fp>(pauli2 * pauli1, control_list), 0.};
    }

    constexpr Fp eps = 1e-12;

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
                              std::uint64_t target) -> std::optional<Gate<Fp>> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I<Fp>();
        if (oct_phase == 4) return gate::Z<Fp>(target, control_list);
        if (oct_phase == 2) return gate::S<Fp>(target, control_list);
        if (oct_phase == 6) return gate::Sdag<Fp>(target, control_list);
        if (oct_phase == 1) return gate::T<Fp>(target, control_list);
        if (oct_phase == 7) return gate::Tdag<Fp>(target, control_list);
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
            Fp phase1 = oct_phase1                   ? oct_phase1.value() * Kokkos::numbers::pi / 4
                        : gate_type1 == GateType::RZ ? RZGate<Fp>(gate1)->angle()
                                                     : U1Gate<Fp>(gate1)->lambda();
            Fp global_phase1 = gate_type1 == GateType::RZ ? -RZGate<Fp>(gate1)->angle() / 2 : 0.;
            Fp phase2 = oct_phase2                   ? oct_phase2.value() * Kokkos::numbers::pi / 4
                        : gate_type2 == GateType::RZ ? RZGate<Fp>(gate2)->angle()
                                                     : U1Gate<Fp>(gate2)->lambda();
            Fp global_phase2 = gate_type2 == GateType::RZ ? -RZGate<Fp>(gate2)->angle() / 2 : 0.;
            Fp global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) {
                return {gate::U1<Fp>(target1, phase1 + phase2, control_list),
                        global_phase1 + global_phase2};
            }
        }
    }

    // Special case: RX
    auto get_rx_angle = [&](Gate<Fp> gate, GateType gate_type) -> std::optional<Fp> {
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::X) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtX) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtXdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RX) return RXGate<Fp>(gate)->angle();
        return std::nullopt;
    };
    auto rx_param1 = get_rx_angle(gate1, gate_type1);
    auto rx_param2 = get_rx_angle(gate2, gate_type2);
    if (rx_param1 && rx_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        Fp global_phase1 = gate_type1 == GateType::RX ? 0. : rx_param1.value() / 2;
        Fp global_phase2 = gate_type2 == GateType::RX ? 0. : rx_param2.value() / 2;
        Fp global_phase = global_phase1 + global_phase2;
        if (target1 == target2) {
            if (std::abs(global_phase) < eps) {
                return {gate::RX<Fp>(target1, rx_param1.value() + rx_param2.value(), control_list),
                        global_phase1 + global_phase2};
            }
        }
    }

    // Special case: RY
    auto get_ry_angle = [&](Gate<Fp> gate, GateType gate_type) -> std::optional<Fp> {
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::Y) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtY) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtYdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RY) return RYGate<Fp>(gate)->angle();
        return std::nullopt;
    };
    auto ry_param1 = get_ry_angle(gate1, gate_type1);
    auto ry_param2 = get_ry_angle(gate2, gate_type2);
    if (ry_param1 && ry_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        Fp global_phase1 = gate_type1 == GateType::RY ? 0. : ry_param1.value() / 2;
        Fp global_phase2 = gate_type2 == GateType::RY ? 0. : ry_param2.value() / 2;
        Fp global_phase = global_phase1 + global_phase2;
        if (target1 == target2) {
            if (std::abs(global_phase) < eps) {
                return {gate::RY<Fp>(target1, ry_param1.value() + ry_param2.value(), control_list),
                        global_phase1 + global_phase2};
            }
        }
    }

    // Special case: Swap duplication
    if (gate_type1 == gate_type2 && gate_type1 == GateType::Swap) {
        if (gate1->target_qubit_mask() == gate2->target_qubit_mask()) return {gate::I<Fp>(), 0.};
    }

    // General case
    return merge_gate_dense_matrix(gate1, gate2);
}
#define FUNC_MACRO(Fp) \
    template std::pair<Gate<Fp>, Fp> merge_gate(const Gate<Fp>&, const Gate<Fp>&);
CALL_MACRO_FOR_FLOAT(FUNC_MACRO)
#undef FUNC_MACRO
}  // namespace scaluq