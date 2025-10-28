#include <scaluq/constant.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/util/utility.hpp>

#include "../prec_space.hpp"

namespace scaluq {
std::pair<Gate<internal::Prec, internal::Space>, double> merge_gate_dense_matrix(
    const Gate<internal::Prec, internal::Space>& gate1,
    const Gate<internal::Prec, internal::Space>& gate2) {
    // 1. 制御ビットが一致する箇所のビットを立てる
    auto common_control_mask = gate1->control_qubit_mask() & gate2->control_qubit_mask();
    // 2. かつ，制御ビットの値も一致する箇所のみビットを立てる
    common_control_mask &= ~(gate1->control_value_mask() ^ gate2->control_value_mask());
    auto common_control_value_mask = common_control_mask & gate1->control_value_mask();
    // 3. 制御ビットが共通しない操作ビットが合成後の操作ビット
    auto merged_operand_mask =
        (gate1->operand_qubit_mask() | gate2->operand_qubit_mask()) & ~common_control_mask;
    auto merged_operand_vector = internal::mask_to_vector(merged_operand_mask);

    auto matrix1 = internal::get_expanded_matrix(gate1->get_matrix(),
                                                 gate1->target_qubit_list(),
                                                 gate1->control_qubit_mask() & ~common_control_mask,
                                                 gate1->control_value_mask() & ~common_control_mask,
                                                 merged_operand_vector);
    auto matrix2 = internal::get_expanded_matrix(gate2->get_matrix(),
                                                 gate2->target_qubit_list(),
                                                 gate2->control_qubit_mask() & ~common_control_mask,
                                                 gate2->control_value_mask() & ~common_control_mask,
                                                 merged_operand_vector);
    auto matrix = matrix2 * matrix1;
    return {gate::DenseMatrix<internal::Prec, internal::Space>(
                merged_operand_vector,
                matrix,
                internal::mask_to_vector(common_control_mask),
                internal::mask_to_vector(common_control_mask, common_control_value_mask)),
            0.};
}

template <>
std::pair<Gate<internal::Prec, internal::Space>, double> merge_gate(
    const Gate<internal::Prec, internal::Space>& gate1,
    const Gate<internal::Prec, internal::Space>& gate2) {
    constexpr double eps = 1e-12;

    GateType gate_type1 = gate1.gate_type();
    GateType gate_type2 = gate2.gate_type();

    if (gate_type1 == GateType::Probabilistic || gate_type2 == GateType::Probabilistic) {
        throw std::runtime_error(
            "merge_gate(const Gate<Prec, Space>&, const Gate<Prec, Space>&): "
            "ProbabilisticGate is not supported.");
    }

    if (gate_type1 == GateType::I) return {gate2, 0.};
    if (gate_type2 == GateType::I) return {gate1, 0.};

    auto gate1_control_mask = gate1->control_qubit_mask();
    auto gate2_control_mask = gate2->control_qubit_mask();

    auto gate1_control_value_mask = gate1->control_value_mask();
    auto gate2_control_value_mask = gate2->control_value_mask();

    if (gate_type1 == GateType::GlobalPhase && gate1_control_mask == 0)
        return {gate2, GlobalPhaseGate<internal::Prec, internal::Space>(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase && gate2_control_mask == 0)
        return {gate1, GlobalPhaseGate<internal::Prec, internal::Space>(gate2)->phase()};

    // 制御ビットとその値がすべて一致しないとき，密行列に直してから合成
    if (!(gate1_control_mask == gate2_control_mask &&
          gate1_control_value_mask == gate2_control_value_mask))
        return merge_gate_dense_matrix(gate1, gate2);
    auto control_qubit_list = gate1->control_qubit_list();
    auto control_value_list = gate1->control_value_list();

    // Special case: Zero qubit
    if (gate_type1 == GateType::GlobalPhase && gate_type2 == GateType::GlobalPhase) {
        return {gate::GlobalPhase<internal::Prec, internal::Space>(
                    GlobalPhaseGate<internal::Prec, internal::Space>(gate1)->phase() +
                        GlobalPhaseGate<internal::Prec, internal::Space>(gate2)->phase(),
                    control_qubit_list,
                    control_value_list),
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
            if (pauli_id1 == pauli_id2) return {gate::I<internal::Prec, internal::Space>(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
                        return {gate::Y<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                Kokkos::numbers::pi / 2};
                    }
                }
            }
            if (pauli_id1 == 2) {
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
                        return {gate::X<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                Kokkos::numbers::pi / 2};
                    }
                }
            }
            if (pauli_id1 == 3) {
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
                        return {gate::Y<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                -Kokkos::numbers::pi / 2};
                    }
                }
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::X<internal::Prec, internal::Space>(
                                    target1, control_qubit_list, control_value_list),
                                Kokkos::numbers::pi / 2};
                    }
                }
            }
        }
    }
    if ((pauli_id1 || gate1.gate_type() == GateType::Pauli) &&
        (pauli_id2 || gate2.gate_type() == GateType::Pauli)) {
        auto pauli1 =
            gate_type1 == GateType::Pauli
                ? PauliGate<internal::Prec, internal::Space>(gate1)->pauli()
                : PauliOperator<internal::Prec, internal::Space>(
                      std::vector{gate1->target_qubit_list()[0]}, std::vector{pauli_id1.value()});
        auto pauli2 =
            gate_type2 == GateType::Pauli
                ? PauliGate<internal::Prec, internal::Space>(gate2)->pauli()
                : PauliOperator<internal::Prec, internal::Space>(
                      std::vector{gate2->target_qubit_list()[0]}, std::vector{pauli_id2.value()});
        return {gate::Pauli<internal::Prec, internal::Space>(
                    pauli2 * pauli1, control_qubit_list, control_value_list),
                0.};
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
    auto oct_phase_gate =
        [&](std::uint64_t oct_phase,
            std::uint64_t target) -> std::optional<Gate<internal::Prec, internal::Space>> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I<internal::Prec, internal::Space>();
        if (oct_phase == 4)
            return gate::Z<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (oct_phase == 2)
            return gate::S<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (oct_phase == 6)
            return gate::Sdag<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (oct_phase == 1)
            return gate::T<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (oct_phase == 7)
            return gate::Tdag<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        return std::nullopt;
    };
    auto oct_phase1 = get_oct_phase(gate_type1);
    auto oct_phase2 = get_oct_phase(gate_type2);
    if (oct_phase1 && oct_phase2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            auto g = oct_phase_gate((oct_phase1.value() + oct_phase2.value()) % 8, target1);
            if (g) return {g.value(), 0.};
        }
    }
    if ((oct_phase1 || gate_type1 == GateType::RZ || gate_type1 == GateType::U1) &&
        (oct_phase2 || gate_type2 == GateType::RZ || gate_type2 == GateType::U1)) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            double phase1 = oct_phase1 ? oct_phase1.value() * Kokkos::numbers::pi / 4
                            : gate_type1 == GateType::RZ
                                ? RZGate<internal::Prec, internal::Space>(gate1)->angle()
                                : U1Gate<internal::Prec, internal::Space>(gate1)->lambda();
            double global_phase1 =
                gate_type1 == GateType::RZ
                    ? -RZGate<internal::Prec, internal::Space>(gate1)->angle() / 2.
                    : 0.;
            double phase2 = oct_phase2 ? oct_phase2.value() * Kokkos::numbers::pi / 4
                            : gate_type2 == GateType::RZ
                                ? RZGate<internal::Prec, internal::Space>(gate2)->angle()
                                : U1Gate<internal::Prec, internal::Space>(gate2)->lambda();
            double global_phase2 =
                gate_type2 == GateType::RZ
                    ? -RZGate<internal::Prec, internal::Space>(gate2)->angle() / 2.
                    : 0.;
            double global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) global_phase = 0.;
            if (gate1_control_mask == 0 || global_phase == 0.) {
                return {gate::U1<internal::Prec, internal::Space>(
                            target1, phase1 + phase2, control_qubit_list, control_value_list),
                        global_phase};
            }
        }
    }

    // Special case: RX
    auto get_quad_rx = [&](GateType gate_type) -> std::optional<std::uint64_t> {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::X) return 2;
        if (gate_type == GateType::SqrtX) return 1;
        if (gate_type == GateType::SqrtXdag) return 3;
        return std::nullopt;
    };
    auto quad_rx_gate =
        [&](std::uint64_t quad_rx,
            std::uint64_t target) -> std::optional<Gate<internal::Prec, internal::Space>> {
        quad_rx &= 3;
        if (quad_rx == 0) return gate::I<internal::Prec, internal::Space>();
        if (quad_rx == 2)
            return gate::X<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (quad_rx == 1)
            return gate::SqrtX<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (quad_rx == 3)
            return gate::SqrtXdag<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        return std::nullopt;
    };
    auto quad_rx1 = get_quad_rx(gate_type1);
    auto quad_rx2 = get_quad_rx(gate_type2);
    if (quad_rx1 && quad_rx2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            auto g = quad_rx_gate((quad_rx1.value() + quad_rx2.value()) % 4, target1);
            if (g) return {g.value(), 0.};
        }
    }
    if ((quad_rx1 || gate_type1 == GateType::RX) && (quad_rx2 || gate_type2 == GateType::RX)) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            double angle1 = quad_rx1 ? quad_rx1.value() * Kokkos::numbers::pi / 2
                                     : RXGate<internal::Prec, internal::Space>(gate1)->angle();
            double global_phase1 = gate_type1 == GateType::RX ? 0. : angle1 / 2.;
            double angle2 = quad_rx2 ? quad_rx2.value() * Kokkos::numbers::pi / 2
                                     : RXGate<internal::Prec, internal::Space>(gate2)->angle();
            double global_phase2 = gate_type2 == GateType::RX ? 0. : angle2 / 2.;
            double global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) global_phase = 0.;
            if (gate1_control_mask == 0 || global_phase == 0.) {
                return {gate::RX<internal::Prec, internal::Space>(
                            target1, angle1 + angle2, control_qubit_list, control_value_list),
                        global_phase};
            }
        }
    }

    // Special case: RY
    auto get_quad_ry = [&](GateType gate_type) -> std::optional<std::uint64_t> {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::Y) return 2;
        if (gate_type == GateType::SqrtY) return 1;
        if (gate_type == GateType::SqrtYdag) return 3;
        return std::nullopt;
    };
    auto quad_ry_gate =
        [&](std::uint64_t quad_ry,
            std::uint64_t target) -> std::optional<Gate<internal::Prec, internal::Space>> {
        quad_ry &= 3;
        if (quad_ry == 0) return gate::I<internal::Prec, internal::Space>();
        if (quad_ry == 2)
            return gate::Y<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (quad_ry == 1)
            return gate::SqrtY<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        if (quad_ry == 3)
            return gate::SqrtYdag<internal::Prec, internal::Space>(
                target, control_qubit_list, control_value_list);
        return std::nullopt;
    };
    auto quad_ry1 = get_quad_ry(gate_type1);
    auto quad_ry2 = get_quad_ry(gate_type2);
    if (quad_ry1 && quad_ry2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            auto g = quad_ry_gate((quad_ry1.value() + quad_ry2.value()) % 4, target1);
            if (g) return {g.value(), 0.};
        }
    }
    if ((quad_ry1 || gate_type1 == GateType::RY) && (quad_ry2 || gate_type2 == GateType::RY)) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        if (target1 == target2) {
            double angle1 = quad_ry1 ? quad_ry1.value() * Kokkos::numbers::pi / 2
                                     : RYGate<internal::Prec, internal::Space>(gate1)->angle();
            double global_phase1 = gate_type1 == GateType::RY ? 0. : angle1 / 2.;
            double angle2 = quad_ry2 ? quad_ry2.value() * Kokkos::numbers::pi / 2
                                     : RYGate<internal::Prec, internal::Space>(gate2)->angle();
            double global_phase2 = gate_type2 == GateType::RY ? 0. : angle2 / 2.;
            double global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) global_phase = 0.;
            if (gate1_control_mask == 0 || global_phase == 0.) {
                return {gate::RY<internal::Prec, internal::Space>(
                            target1, angle1 + angle2, control_qubit_list, control_value_list),
                        global_phase};
            }
        }
    }

    // Special case: Swap duplication
    if (gate_type1 == gate_type2 && gate_type1 == GateType::Swap) {
        if (gate1->target_qubit_mask() == gate2->target_qubit_mask())
            return {gate::I<internal::Prec, internal::Space>(), 0.};
    }

    // General case
    return merge_gate_dense_matrix(gate1, gate2);
}
}  // namespace scaluq
