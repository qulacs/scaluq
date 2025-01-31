#include <scaluq/constant.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/util/utility.hpp>

#include "../util/template.hpp"

namespace scaluq {
<<<<<<< HEAD
template <Precision Prec>
std::pair<Gate<Prec>, double> merge_gate_dense_matrix(const Gate<Prec>& gate1,
                                                      const Gate<Prec>& gate2) {
=======
FLOAT_AND_SPACE(Fp, Sp)
std::pair<Gate<Fp, Sp>, Fp> merge_gate_dense_matrix(const Gate<Fp, Sp>& gate1,
                                                    const Gate<Fp, Sp>& gate2) {
>>>>>>> set-space
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
    auto matrix = matrix2 * matrix1;
<<<<<<< HEAD
    std::cerr << matrix << std::endl;
    return {gate::DenseMatrix<Prec>(
=======
    return {gate::DenseMatrix<Fp, Sp>(
>>>>>>> set-space
                merged_operand_vector, matrix, internal::mask_to_vector(common_control_mask)),
            0.};
}

<<<<<<< HEAD
template <Precision Prec>
std::pair<Gate<Prec>, double> merge_gate(const Gate<Prec>& gate1, const Gate<Prec>& gate2) {
    constexpr double eps = 1e-12;

=======
FLOAT_AND_SPACE(Fp, Sp)
std::pair<Gate<Fp, Sp>, Fp> merge_gate(const Gate<Fp, Sp>& gate1, const Gate<Fp, Sp>& gate2) {
>>>>>>> set-space
    GateType gate_type1 = gate1.gate_type();
    GateType gate_type2 = gate2.gate_type();

    if (gate_type1 == GateType::Probablistic || gate_type2 == GateType::Probablistic) {
        throw std::runtime_error(
<<<<<<< HEAD
            "merge_gate(const Gate<Prec>&, const Gate<Prec>&): ProbablisticGate is not supported.");
=======
            "merge_gate(const Gate<Fp, Sp>&, const Gate<Fp, Sp>&): ProbablisticGate is not "
            "supported.");
>>>>>>> set-space
    }

    if (gate_type1 == GateType::I) return {gate2, 0.};
    if (gate_type2 == GateType::I) return {gate1, 0.};

    auto gate1_control_mask = gate1->control_qubit_mask();
    auto gate2_control_mask = gate2->control_qubit_mask();

    if (gate_type1 == GateType::GlobalPhase && gate1_control_mask == 0)
<<<<<<< HEAD
        return {gate2, GlobalPhaseGate<Prec>(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase && gate2_control_mask == 0)
        return {gate1, GlobalPhaseGate<Prec>(gate2)->phase()};
=======
        return {gate2, GlobalPhaseGate<Fp, Sp>(gate1)->phase()};
    if (gate_type2 == GateType::GlobalPhase && gate2_control_mask == 0)
        return {gate1, GlobalPhaseGate<Fp, Sp>(gate2)->phase()};
>>>>>>> set-space

    if (gate1_control_mask != gate2_control_mask) return merge_gate_dense_matrix(gate1, gate2);
    auto control_list = internal::mask_to_vector(gate1_control_mask);

    // Special case: Zero qubit
    if (gate_type1 == GateType::GlobalPhase && gate_type2 == GateType::GlobalPhase) {
<<<<<<< HEAD
        return {gate::GlobalPhase<Prec>(
                    GlobalPhaseGate<Prec>(gate1)->phase() + GlobalPhaseGate<Prec>(gate2)->phase(),
                    control_list),
=======
        return {gate::GlobalPhase<Fp, Sp>(GlobalPhaseGate<Fp, Sp>(gate1)->phase() +
                                              GlobalPhaseGate<Fp, Sp>(gate2)->phase(),
                                          control_list),
>>>>>>> set-space
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
<<<<<<< HEAD
            if (pauli_id1 == pauli_id2) return {gate::I<Prec>(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<Prec>(target1, control_list), -Kokkos::numbers::pi / 2};
=======
            if (pauli_id1 == pauli_id2) return {gate::I<Fp, Sp>(), 0.};
            if (pauli_id1 == 1) {
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
                        return {gate::Z<Fp, Sp>(target1, control_list), -Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
<<<<<<< HEAD
                        return {gate::Y<Prec>(target1, control_list), Kokkos::numbers::pi / 2};
=======
                        return {gate::Y<Fp, Sp>(target1, control_list), Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
            }
            if (pauli_id1 == 2) {
                if (pauli_id2 == 3) {
                    if (gate1_control_mask == 0) {
<<<<<<< HEAD
                        return {gate::X<Prec>(target1, control_list), -Kokkos::numbers::pi / 2};
=======
                        return {gate::X<Fp, Sp>(target1, control_list), -Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
<<<<<<< HEAD
                        return {gate::Z<Prec>(target1, control_list), Kokkos::numbers::pi / 2};
=======
                        return {gate::Z<Fp, Sp>(target1, control_list), Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
            }
            if (pauli_id1 == 3) {
                if (pauli_id2 == 1) {
                    if (gate1_control_mask == 0) {
<<<<<<< HEAD
                        return {gate::Y<Prec>(target1, control_list), -Kokkos::numbers::pi / 2};
=======
                        return {gate::Y<Fp, Sp>(target1, control_list), -Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
                if (pauli_id2 == 2) {
                    if (gate1_control_mask == 0) {
<<<<<<< HEAD
                        return {gate::X<Prec>(target1, control_list), Kokkos::numbers::pi / 2};
=======
                        return {gate::X<Fp, Sp>(target1, control_list), Kokkos::numbers::pi / 2};
>>>>>>> set-space
                    }
                }
            }
        }
    }
    if ((pauli_id1 || gate1.gate_type() == GateType::Pauli) &&
        (pauli_id2 || gate2.gate_type() == GateType::Pauli)) {
        auto pauli1 = gate_type1 == GateType::Pauli
<<<<<<< HEAD
                          ? PauliGate<Prec>(gate1)->pauli()
                          : PauliOperator<Prec>(std::vector{gate1->target_qubit_list()[0]},
                                                std::vector{pauli_id1.value()});
        auto pauli2 = gate_type2 == GateType::Pauli
                          ? PauliGate<Prec>(gate2)->pauli()
                          : PauliOperator<Prec>(std::vector{gate2->target_qubit_list()[0]},
                                                std::vector{pauli_id2.value()});
        return {gate::Pauli<Prec>(pauli2 * pauli1, control_list), 0.};
=======
                          ? PauliGate<Fp, Sp>(gate1)->pauli()
                          : PauliOperator<Fp, Sp>(std::vector{gate1->target_qubit_list()[0]},
                                                  std::vector{pauli_id1.value()});
        auto pauli2 = gate_type2 == GateType::Pauli
                          ? PauliGate<Fp, Sp>(gate2)->pauli()
                          : PauliOperator<Fp, Sp>(std::vector{gate2->target_qubit_list()[0]},
                                                  std::vector{pauli_id2.value()});
        return {gate::Pauli<Fp, Sp>(pauli2 * pauli1, control_list), 0.};
>>>>>>> set-space
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
<<<<<<< HEAD
                              std::uint64_t target) -> std::optional<Gate<Prec>> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I<Prec>();
        if (oct_phase == 4) return gate::Z<Prec>(target, control_list);
        if (oct_phase == 2) return gate::S<Prec>(target, control_list);
        if (oct_phase == 6) return gate::Sdag<Prec>(target, control_list);
        if (oct_phase == 1) return gate::T<Prec>(target, control_list);
        if (oct_phase == 7) return gate::Tdag<Prec>(target, control_list);
=======
                              std::uint64_t target) -> std::optional<Gate<Fp, Sp>> {
        oct_phase &= 7;
        if (oct_phase == 0) return gate::I<Fp, Sp>();
        if (oct_phase == 4) return gate::Z<Fp, Sp>(target, control_list);
        if (oct_phase == 2) return gate::S<Fp, Sp>(target, control_list);
        if (oct_phase == 6) return gate::Sdag<Fp, Sp>(target, control_list);
        if (oct_phase == 1) return gate::T<Fp, Sp>(target, control_list);
        if (oct_phase == 7) return gate::Tdag<Fp, Sp>(target, control_list);
>>>>>>> set-space
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
<<<<<<< HEAD
            double phase1 = oct_phase1 ? oct_phase1.value() * Kokkos::numbers::pi / 4
                            : gate_type1 == GateType::RZ ? RZGate<Prec>(gate1)->angle()
                                                         : U1Gate<Prec>(gate1)->lambda();
            double global_phase1 =
                gate_type1 == GateType::RZ ? -RZGate<Prec>(gate1)->angle() / 2. : 0.;
            double phase2 = oct_phase2 ? oct_phase2.value() * Kokkos::numbers::pi / 4
                            : gate_type2 == GateType::RZ ? RZGate<Prec>(gate2)->angle()
                                                         : U1Gate<Prec>(gate2)->lambda();
            double global_phase2 =
                gate_type2 == GateType::RZ ? -RZGate<Prec>(gate2)->angle() / 2. : 0.;
            double global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) {
                return {gate::U1<Prec>(target1, phase1 + phase2, control_list),
=======
            Fp phase1 = oct_phase1                   ? oct_phase1.value() * Kokkos::numbers::pi / 4
                        : gate_type1 == GateType::RZ ? RZGate<Fp, Sp>(gate1)->angle()
                                                     : U1Gate<Fp, Sp>(gate1)->lambda();
            Fp global_phase1 =
                gate_type1 == GateType::RZ ? -RZGate<Fp, Sp>(gate1)->angle() / 2 : 0.;
            Fp phase2 = oct_phase2                   ? oct_phase2.value() * Kokkos::numbers::pi / 4
                        : gate_type2 == GateType::RZ ? RZGate<Fp, Sp>(gate2)->angle()
                                                     : U1Gate<Fp, Sp>(gate2)->lambda();
            Fp global_phase2 =
                gate_type2 == GateType::RZ ? -RZGate<Fp, Sp>(gate2)->angle() / 2 : 0.;
            Fp global_phase = global_phase1 + global_phase2;
            if (std::abs(global_phase) < eps) {
                return {gate::U1<Fp, Sp>(target1, phase1 + phase2, control_list),
>>>>>>> set-space
                        global_phase1 + global_phase2};
            }
        }
    }

    // Special case: RX
<<<<<<< HEAD
    auto get_rx_angle = [&](Gate<Prec> gate, GateType gate_type) -> std::optional<double> {
        if (gate_type == GateType::I) return 0;
        if (gate_type == GateType::X) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtX) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtXdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RX) return RXGate<Prec>(gate)->angle();
=======
    auto get_rx_angle = [&](Gate<Fp, Sp> gate, GateType gate_type) -> std::optional<Fp> {
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::X) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtX) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtXdag) return -Kokkos::numbers::pi / 2;
        if (gate_type == GateType::RX) return RXGate<Fp, Sp>(gate)->angle();
>>>>>>> set-space
        return std::nullopt;
    };
    auto rx_param1 = get_rx_angle(gate1, gate_type1);
    auto rx_param2 = get_rx_angle(gate2, gate_type2);
    if (rx_param1 && rx_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        double global_phase1 = gate_type1 == GateType::RX ? 0. : rx_param1.value() / 2.;
        double global_phase2 = gate_type2 == GateType::RX ? 0. : rx_param2.value() / 2.;
        double global_phase = global_phase1 + global_phase2;
        if (target1 == target2) {
            if (std::abs(global_phase) < eps) {
                return {
<<<<<<< HEAD
                    gate::RX<Prec>(target1, rx_param1.value() + rx_param2.value(), control_list),
=======
                    gate::RX<Fp, Sp>(target1, rx_param1.value() + rx_param2.value(), control_list),
>>>>>>> set-space
                    global_phase1 + global_phase2};
            }
        }
    }

    // Special case: RY
<<<<<<< HEAD
    auto get_ry_angle = [&](Gate<Prec> gate, GateType gate_type) -> std::optional<double> {
=======
    auto get_ry_angle = [&](Gate<Fp, Sp> gate, GateType gate_type) -> std::optional<Fp> {
>>>>>>> set-space
        if (gate_type == GateType::I) return 0.;
        if (gate_type == GateType::Y) return Kokkos::numbers::pi;
        if (gate_type == GateType::SqrtY) return Kokkos::numbers::pi / 2;
        if (gate_type == GateType::SqrtYdag) return -Kokkos::numbers::pi / 2;
<<<<<<< HEAD
        if (gate_type == GateType::RY) return RYGate<Prec>(gate)->angle();
=======
        if (gate_type == GateType::RY) return RYGate<Fp, Sp>(gate)->angle();
>>>>>>> set-space
        return std::nullopt;
    };
    auto ry_param1 = get_ry_angle(gate1, gate_type1);
    auto ry_param2 = get_ry_angle(gate2, gate_type2);
    if (ry_param1 && ry_param2) {
        std::uint64_t target1 = gate1->target_qubit_list()[0];
        std::uint64_t target2 = gate2->target_qubit_list()[0];
        double global_phase1 = gate_type1 == GateType::RY ? 0. : ry_param1.value() / 2.;
        double global_phase2 = gate_type2 == GateType::RY ? 0. : ry_param2.value() / 2.;
        double global_phase = global_phase1 + global_phase2;
        if (target1 == target2) {
            if (std::abs(global_phase) < eps) {
                return {
<<<<<<< HEAD
                    gate::RY<Prec>(target1, ry_param1.value() + ry_param2.value(), control_list),
=======
                    gate::RY<Fp, Sp>(target1, ry_param1.value() + ry_param2.value(), control_list),
>>>>>>> set-space
                    global_phase1 + global_phase2};
            }
        }
    }

    // Special case: Swap duplication
    if (gate_type1 == gate_type2 && gate_type1 == GateType::Swap) {
<<<<<<< HEAD
        if (gate1->target_qubit_mask() == gate2->target_qubit_mask()) return {gate::I<Prec>(), 0.};
=======
        if (gate1->target_qubit_mask() == gate2->target_qubit_mask())
            return {gate::I<Fp, Sp>(), 0.};
>>>>>>> set-space
    }

    // General case
    return merge_gate_dense_matrix(gate1, gate2);
}
<<<<<<< HEAD
#define FUNC_MACRO(Prec) \
    template std::pair<Gate<Prec>, double> merge_gate(const Gate<Prec>&, const Gate<Prec>&);
SCALUQ_CALL_MACRO_FOR_PRECISION(FUNC_MACRO)
=======
#define FUNC_MACRO(Fp, Sp) \
    template std::pair<Gate<Fp, Sp>, Fp> merge_gate(const Gate<Fp, Sp>&, const Gate<Fp, Sp>&);
CALL_MACRO_FOR_FLOAT_AND_SPACE(FUNC_MACRO)
>>>>>>> set-space
#undef FUNC_MACRO
}  // namespace scaluq
