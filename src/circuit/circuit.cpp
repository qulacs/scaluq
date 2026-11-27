#include <array>
#include <limits>
#include <numeric>
#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/gate/param_gate_factory.hpp>
#include <scaluq/util/utility.hpp>
#include <span>

namespace scaluq {
namespace {
template <Precision Prec>
void check_gradient_circuit(const Circuit<Prec>& circuit,
                            const std::map<std::string, double>& parameters) {
    constexpr double tolerance = Prec == Precision::F64   ? 1e-12
                                 : Prec == Precision::F32 ? 1e-5
                                 : Prec == Precision::F16 ? 0.005
                                                          : 0.02;
    for (std::uint64_t i = 0; i < circuit.n_gates(); ++i) {
        if (circuit.get_classical_condition_at(i)) {
            throw std::invalid_argument(
                "Expectation gradient does not support classical conditions.");
        }
        const auto& entry = circuit.get_gate_at(i);
        if (entry.index() == 1) {
            const auto& [gate, key] = std::get<1>(entry);
            const auto value = parameters.find(key);
            if (value == parameters.end() || !std::isfinite(value->second) ||
                !std::isfinite(gate->param_coef()) ||
                !std::isfinite(gate->param_coef() * value->second)) {
                throw std::invalid_argument(
                    "Expectation gradient requires finite parameters and coefficients.");
            }
            switch (gate.param_gate_type()) {
                case ParamGateType::ParamRX:
                case ParamGateType::ParamRY:
                case ParamGateType::ParamRZ:
                    break;
                case ParamGateType::ParamPauliRotation: {
                    const auto coef = ParamPauliRotationGate<Prec>(gate)->pauli().coef();
                    if (!std::isfinite(coef.real()) || coef.imag() != 0.) {
                        throw std::invalid_argument(
                            "Expectation gradient requires Hermitian rotation generators.");
                    }
                    break;
                }
                default:
                    throw std::invalid_argument(
                        "Expectation gradient requires deterministic unitary gates.");
            }
            continue;
        }
        const auto& gate = std::get<0>(entry);
        switch (gate.gate_type()) {
            case GateType::Pauli: {
                const double norm = std::norm(PauliGate<Prec>(gate)->pauli().coef());
                if (!std::isfinite(norm) || std::abs(norm - 1.) > tolerance) {
                    throw std::invalid_argument(
                        "Expectation gradient requires unitary Pauli gates.");
                }
                continue;
            }
            case GateType::PauliRotation: {
                const auto rotation = PauliRotationGate<Prec>(gate);
                const auto coef = rotation->pauli().coef();
                if (!std::isfinite(coef.real()) || coef.imag() != 0. ||
                    !std::isfinite(rotation->angle())) {
                    throw std::invalid_argument(
                        "Expectation gradient requires unitary Pauli rotations.");
                }
                continue;
            }
            case GateType::P0:
            case GateType::P1:
            case GateType::Measurement:
            case GateType::Probabilistic:
            case GateType::SparseMatrix:
                throw std::invalid_argument(
                    "Expectation gradient requires deterministic gates with unitary inverses.");
            case GateType::DenseMatrix:
            case GateType::Unknown:
            case GateType::GlobalPhase:
            case GateType::RX:
            case GateType::RY:
            case GateType::RZ:
            case GateType::U1:
            case GateType::U2:
            case GateType::U3:
                break;
            default:
                continue;
        }
        const auto matrix = gate->get_matrix();
        if (matrix.rows() != matrix.cols()) {
            throw std::invalid_argument("Expectation gradient requires square unitary matrices.");
        }
        const auto gram = (matrix.adjoint() * matrix).eval();
        for (decltype(gram.rows()) row = 0; row < gram.rows(); ++row) {
            for (decltype(gram.cols()) col = 0; col < gram.cols(); ++col) {
                const double error = std::abs(gram(row, col) - StdComplex(row == col ? 1. : 0.));
                if (!std::isfinite(error) || error > tolerance) {
                    throw std::invalid_argument("Expectation gradient requires unitary matrices.");
                }
            }
        }
    }
}
}  // namespace

template <Precision Prec>
std::set<std::string> Circuit<Prec>::key_set() const {
    std::set<std::string> key_set;
    for (auto&& gate : _gate_list) {
        if (gate.index() == 1) key_set.insert(std::get<1>(gate).second);
    }
    return key_set;
}

template <Precision Prec>
std::uint64_t Circuit<Prec>::calculate_depth() const {
    const std::uint64_t n_qubits = required_n_qubits();
    if (n_qubits == 0) return 0;
    std::vector<std::uint64_t> filled_step(n_qubits, 0ULL);
    for (const auto& gate : _gate_list) {
        std::vector<std::uint64_t> control_qubits =
            gate.index() == 0 ? std::get<0>(gate)->control_qubit_list()
                              : std::get<1>(gate).first->control_qubit_list();
        std::vector<std::uint64_t> target_qubits =
            gate.index() == 0 ? std::get<0>(gate)->target_qubit_list()
                              : std::get<1>(gate).first->target_qubit_list();
        std::uint64_t max_step_amount_target_qubits = 0;
        for (std::uint64_t control : control_qubits) {
            if (max_step_amount_target_qubits < filled_step[control]) {
                max_step_amount_target_qubits = filled_step[control];
            }
        }
        for (std::uint64_t target : target_qubits) {
            if (max_step_amount_target_qubits < filled_step[target]) {
                max_step_amount_target_qubits = filled_step[target];
            }
        }
        for (std::uint64_t control : control_qubits) {
            filled_step[control] = max_step_amount_target_qubits + 1;
        }
        for (std::uint64_t target : target_qubits) {
            filled_step[target] = max_step_amount_target_qubits + 1;
        }
    }
    return *std::ranges::max_element(filled_step);
}

template <Precision Prec>
void Circuit<Prec>::add_circuit(const Circuit<Prec>& circuit) {
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    _classical_conditions.reserve(_classical_conditions.size() +
                                  circuit._classical_conditions.size());
    for (const auto& gate : circuit._gate_list) {
        _gate_list.push_back(gate);
    }
    for (const auto& condition : circuit._classical_conditions) {
        _classical_conditions.push_back(condition);
    }
}
template <Precision Prec>
// The circuit is consumed by moving its elements individually.
// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
void Circuit<Prec>::add_circuit(Circuit<Prec>&& circuit) {
    _gate_list.reserve(_gate_list.size() + circuit._gate_list.size());
    _classical_conditions.reserve(_classical_conditions.size() +
                                  circuit._classical_conditions.size());
    for (auto&& gate : circuit._gate_list) {
        _gate_list.push_back(std::move(gate));
    }
    for (auto&& condition : circuit._classical_conditions) {
        _classical_conditions.push_back(std::move(condition));
    }
}

template <Precision Prec>
template <ExecutionSpace Space>
void Circuit<Prec>::update_quantum_state(StateVector<Prec, Space>& state,
                                         const std::map<std::string, double>& parameters,
                                         std::optional<std::uint64_t> seed) const {
    if (has_classical_instructions()) {
        throw std::runtime_error(
            "Circuit::update_quantum_state(StateVector&, ...): a classical register is required "
            "when the circuit contains measurement or classically controlled instructions.");
    }
    ClassicalRegister classical_register(0);
    update_quantum_state(state, classical_register, parameters, seed);
}

template <Precision Prec>
template <ExecutionSpace Space>
void Circuit<Prec>::update_quantum_state(StateVector<Prec, Space>& state,
                                         ClassicalRegister& classical_register,
                                         const std::map<std::string, double>& parameters,
                                         std::optional<std::uint64_t> seed) const {
    check_state_vector_n_qubits(state.n_qubits());
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) continue;
        const auto& key = std::get<1>(gate).second;
        if (!parameters.contains(key)) {
            using namespace std::string_literals;
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVector&, const std::map<std::string, double>&) const: parameter named "s +
                std::string(key) + "is not given.");
        }
    }
    std::mt19937_64 random_engine(internal::resolve_seed(seed));
    internal::ExecutionContext<Prec, Space> context{state, classical_register, random_engine};
    for (std::uint64_t idx = 0; idx < _gate_list.size(); ++idx) {
        const auto& classical_condition = _classical_conditions[idx];
        if (classical_condition && !(*classical_condition)(classical_register)) {
            continue;
        }
        auto&& gate = _gate_list[idx];
        if (gate.index() == 0) {
            const auto& fixed_gate = std::get<0>(gate);
            fixed_gate->update_quantum_state(context);
        } else {
            const auto& [param_gate, key] = std::get<1>(gate);
            param_gate->update_quantum_state(context, parameters.at(key));
        }
    }
}

template <Precision Prec>
template <ExecutionSpace Space>
void Circuit<Prec>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const {
    if (has_classical_instructions()) {
        throw std::runtime_error(
            "Circuit::update_quantum_state(StateVectorBatched&, ...): a classical register is "
            "required when the circuit contains measurement or classically controlled "
            "instructions.");
    }
    ClassicalRegisterBatched classical_register(0, states.batch_size());
    update_quantum_state(states, classical_register, parameters, seed);
}

template <Precision Prec>
template <ExecutionSpace Space>
void Circuit<Prec>::update_quantum_state(
    StateVectorBatched<Prec, Space>& states,
    ClassicalRegisterBatched& classical_register,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const {
    check_state_vector_n_qubits(states.n_qubits());
    if (classical_register.batch_size() != states.batch_size()) {
        throw std::runtime_error(
            "Circuit::update_quantum_state(StateVectorBatched&, ClassicalRegisterBatched&, ...): "
            "batch size mismatch.");
    }
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0) continue;
        const auto& key = std::get<1>(gate).second;
        if (!parameters.contains(key)) {
            using namespace std::string_literals;
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVector&, const std::map<std::string_view, double>&) const: parameter named "s +
                std::string(key) + "is not given.");
        }
        if (parameters.at(key).size() != states.batch_size()) {
            throw std::runtime_error(
                "Circuit::update_quantum_state(StateVectorBatched&, const std::map<std::string, "
                "std::vector<double>>&): parameter size mismatch.");
        }
    }
    std::mt19937_64 random_engine(internal::resolve_seed(seed));
    internal::ExecutionContextBatched<Prec, Space> context{
        states, classical_register, random_engine};
    for (std::uint64_t idx = 0; idx < _gate_list.size(); ++idx) {
        auto&& gate = _gate_list[idx];
        if (_classical_conditions[idx].has_value()) {
            for (std::uint64_t batch_index = 0; batch_index < states.batch_size(); ++batch_index) {
                auto& reg = classical_register[batch_index];
                if (!_classical_conditions[idx].value()(reg)) continue;
                auto state = states.view_state_vector_at(batch_index);
                internal::ExecutionContext<Prec, Space> state_context{state, reg, random_engine};
                if (gate.index() == 0) {
                    std::get<0>(gate)->update_quantum_state(state_context);
                } else {
                    const auto& [param_gate, key] = std::get<1>(gate);
                    param_gate->update_quantum_state(state_context,
                                                     parameters.at(key)[batch_index]);
                }
            }
            continue;
        }
        if (gate.index() == 0) {
            const auto& fixed_gate = std::get<0>(gate);
            fixed_gate->update_quantum_state(context);
        } else {
            const auto& [param_gate, key] = std::get<1>(gate);
            param_gate->update_quantum_state(context, parameters.at(key));
        }
    }
}

template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Host>(
    StateVector<internal::Prec, ExecutionSpace::Host>& state,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Host>(
    StateVector<internal::Prec, ExecutionSpace::Host>& state,
    ClassicalRegister& classical_register,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Host>(
    StateVectorBatched<internal::Prec, ExecutionSpace::Host>& states,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Host>(
    StateVectorBatched<internal::Prec, ExecutionSpace::Host>& states,
    ClassicalRegisterBatched& classical_register,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;

template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::HostSerial>(
    StateVector<internal::Prec, ExecutionSpace::HostSerial>& state,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::HostSerial>(
    StateVector<internal::Prec, ExecutionSpace::HostSerial>& state,
    ClassicalRegister& classical_register,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::HostSerial>(
    StateVectorBatched<internal::Prec, ExecutionSpace::HostSerial>& states,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::HostSerial>(
    StateVectorBatched<internal::Prec, ExecutionSpace::HostSerial>& states,
    ClassicalRegisterBatched& classical_register,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;

#ifdef SCALUQ_USE_DEVICE
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Default>(
    StateVector<internal::Prec, ExecutionSpace::Default>& state,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Default>(
    StateVector<internal::Prec, ExecutionSpace::Default>& state,
    ClassicalRegister& classical_register,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Default>(
    StateVectorBatched<internal::Prec, ExecutionSpace::Default>& states,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
template void Circuit<internal::Prec>::update_quantum_state<ExecutionSpace::Default>(
    StateVectorBatched<internal::Prec, ExecutionSpace::Default>& states,
    ClassicalRegisterBatched& classical_register,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
#endif  // SCALUQ_USE_DEVICE

template <Precision Prec>
Circuit<Prec> Circuit<Prec>::copy() const {
    Circuit<Prec> ccircuit;
    ccircuit._gate_list.reserve(_gate_list.size());
    for (auto&& gate : _gate_list) {
        if (gate.index() == 0)
            ccircuit._gate_list.push_back(std::get<0>(gate));
        else {
            const auto& [param_gate, key] = std::get<1>(gate);
            ccircuit._gate_list.push_back(std::make_pair(param_gate, key));
        }
    }
    ccircuit._classical_conditions = _classical_conditions;
    return ccircuit;
}

template <Precision Prec>
Circuit<Prec> Circuit<Prec>::get_inverse() const {
    Circuit<Prec> icircuit;
    icircuit._gate_list.reserve(_gate_list.size());
    icircuit._classical_conditions.reserve(_classical_conditions.size());
    for (std::uint64_t idx = 0; idx < _gate_list.size(); ++idx) {
        auto&& gate = _gate_list[_gate_list.size() - 1 - idx];
        if (gate.index() == 0)
            icircuit._gate_list.push_back(std::get<0>(gate)->get_inverse());
        else {
            const auto& [param_gate, key] = std::get<1>(gate);
            icircuit._gate_list.push_back(std::make_pair(param_gate->get_inverse(), key));
        }
        icircuit._classical_conditions.push_back(
            _classical_conditions[_classical_conditions.size() - 1 - idx]);
    }
    return icircuit;
}

template <Precision Prec>
template <ExecutionSpace Space>
void Circuit<Prec>::optimize(std::uint64_t max_block_size) {
    std::vector<GateWithKey> new_gate_list;  // result
    std::vector<std::optional<ClassicalCondition>> new_classical_conditions;
    double global_phase = 0.;
    std::vector<Gate<Prec>> gate_pool;  // waitlist for merge or push
    constexpr std::uint64_t NO_GATES = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t n_qubits = required_n_qubits();
    std::vector<std::uint64_t> waiting_gate_idx_at_qubit(
        n_qubits,
        NO_GATES);  // which gate is waiting in the qubit (by index of gate_pool)
    auto get_operand_list = [&](const GateWithKey& gate_with_key) {
        if (gate_with_key.index() == 0)
            return std::get<0>(gate_with_key)->operand_qubit_list();
        else
            return std::get<1>(gate_with_key).first->operand_qubit_list();
    };
    auto push = [&](const GateWithKey& gate_with_key,
                    std::optional<ClassicalCondition> condition = std::nullopt) {
        // clear waitlist and push gate to result
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            waiting_gate_idx_at_qubit[operand] = NO_GATES;
        }
        new_gate_list.push_back(gate_with_key);
        new_classical_conditions.push_back(std::move(condition));
    };
    auto wait = [&](const Gate<Prec>& gate) {
        // wait for merge or push
        std::uint64_t new_idx = gate_pool.size();
        for (std::uint64_t operand : gate->operand_qubit_list()) {
            waiting_gate_idx_at_qubit[operand] = new_idx;
        }
        gate_pool.push_back(gate);
    };
    auto get_waiting_gate_indices = [&](const GateWithKey& gate_with_key) {
        std::vector<std::uint64_t> waiting_gate_indices;
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            if (waiting_gate_idx_at_qubit[operand] != NO_GATES)
                waiting_gate_indices.push_back(waiting_gate_idx_at_qubit[operand]);
        }
        std::ranges::sort(waiting_gate_indices);
        waiting_gate_indices.erase(std::ranges::unique(waiting_gate_indices).begin(),
                                   waiting_gate_indices.end());
        return waiting_gate_indices;
    };
    auto push_waiting_gates = [&](const GateWithKey& gate_with_key) {
        for (std::uint64_t idx : get_waiting_gate_indices(gate_with_key)) {
            push(gate_pool[idx]);
        }
    };
    auto get_newly_applied_qubits = [&](const GateWithKey& gate_with_key) {
        std::vector<std::uint64_t> newly_applied_qubits;
        for (std::uint64_t operand : get_operand_list(gate_with_key)) {
            if (waiting_gate_idx_at_qubit[operand] == NO_GATES)
                newly_applied_qubits.push_back(operand);
        }
        return newly_applied_qubits;
    };

    for (std::uint64_t idx = 0; idx < _gate_list.size(); ++idx) {
        const GateWithKey& gate_with_key = _gate_list[idx];
        const auto& classical_condition = _classical_conditions[idx];
        if (classical_condition.has_value() || gate_with_key.index() == 1 ||
            std::get<0>(gate_with_key).gate_type() == GateType::Probabilistic ||
            std::get<0>(gate_with_key).gate_type() == GateType::Measurement) {
            // ParamGate, Probabilistic, Gate with classical branches, and Measurement cannot be
            // merged with others
            push_waiting_gates(gate_with_key);
            push(gate_with_key, classical_condition);
            continue;
        }
        const Gate<Prec>& gate = std::get<0>(gate_with_key);
        if (gate.gate_type() == GateType::I) continue;  // IGate is ignored
        if (gate.gate_type() == GateType::GlobalPhase && gate->control_qubit_mask() == 0ULL) {
            // non-controlled GlobalPhase is ignored
            global_phase += GlobalPhaseGate<Prec>(gate)->phase();
            continue;
        }
        std::vector<std::uint64_t> waiting_gate_indices = get_waiting_gate_indices(gate_with_key);
        if (waiting_gate_indices.empty()) {
            // just wait
            wait(gate);
            continue;
        }
        std::vector<std::uint64_t> newly_applied_qubits = get_newly_applied_qubits(gate_with_key);
        std::uint64_t new_gate_size =
            std::accumulate(waiting_gate_indices.begin(),
                            waiting_gate_indices.end(),
                            newly_applied_qubits.size(),
                            [&](std::uint64_t acc, std::uint64_t idx) {
                                return acc + gate_pool[idx]->operand_qubit_list().size();
                            });
        auto is_pauli = [&](const Gate<Prec>& gate) {
            return gate.gate_type() == GateType::X || gate.gate_type() == GateType::Y ||
                   gate.gate_type() == GateType::Z || gate.gate_type() == GateType::Pauli;
        };
        auto is_pure_pauli = [&](const Gate<Prec>& gate) {
            return gate->control_qubit_mask() == 0ULL && is_pauli(gate);
        };
        bool all_pauli = is_pure_pauli(gate) &&
                         std::ranges::all_of(waiting_gate_indices, [&](std::uint64_t idx) {
                             return is_pure_pauli(gate_pool[idx]);
                         });
        if (waiting_gate_indices.size() == 1) {
            const Gate<Prec>& previous_gate = gate_pool[waiting_gate_indices[0]];
            // common control qubits are not counted as size
            std::uint64_t control_qubit_mask1 = gate->control_qubit_mask();
            std::uint64_t control_value_mask1 = gate->control_value_mask();
            std::uint64_t control_qubit_mask2 = previous_gate->control_qubit_mask();
            std::uint64_t control_value_mask2 = previous_gate->control_value_mask();
            new_gate_size -= std::popcount(control_qubit_mask1 & control_qubit_mask2 &
                                           ~(control_value_mask1 ^ control_value_mask2));

            // check whether both gates are pauli with same control qubits
            if (control_qubit_mask1 == control_qubit_mask2 &&
                control_value_mask1 == control_value_mask2 && is_pauli(gate) &&
                is_pauli(previous_gate)) {
                all_pauli = true;
            }
        }
        if (all_pauli || new_gate_size <= max_block_size) {
            // merge with waiting gates
            Gate<Prec> merged_gate = gate;
            for (std::uint64_t idx : waiting_gate_indices) {
                const auto& [new_merged_gate, phase] =
                    merge_gate<Prec, Space>(gate_pool[idx], merged_gate);
                merged_gate = new_merged_gate;
                global_phase += phase;
                // Cancellation can remove operands from the merged gate.
                for (std::uint64_t operand : gate_pool[idx]->operand_qubit_list()) {
                    waiting_gate_idx_at_qubit[operand] = NO_GATES;
                }
            }

            // wait
            wait(merged_gate);
        } else {
            // not merge
            // push waiting gates
            for (std::uint64_t idx : waiting_gate_indices) {
                push(gate_pool[idx]);
            }

            // wait
            wait(gate);
        }
    }
    std::vector<std::uint64_t> finally_waiting_gate_indices;
    for (std::uint64_t idx : waiting_gate_idx_at_qubit) {
        if (idx != NO_GATES) finally_waiting_gate_indices.push_back(idx);
    }
    std::ranges::sort(finally_waiting_gate_indices);
    finally_waiting_gate_indices.erase(std::ranges::unique(finally_waiting_gate_indices).begin(),
                                       finally_waiting_gate_indices.end());
    for (std::uint64_t idx : finally_waiting_gate_indices) {
        new_gate_list.push_back(gate_pool[idx]);
        new_classical_conditions.push_back(std::nullopt);
    }
    if (std::abs(global_phase) > 1e-12) push(gate::GlobalPhase<Prec>(global_phase));
    _gate_list.swap(new_gate_list);
    _classical_conditions.swap(new_classical_conditions);
}
template void Circuit<internal::Prec>::optimize<ExecutionSpace::Host>(std::uint64_t max_block_size);
template void Circuit<internal::Prec>::optimize<ExecutionSpace::HostSerial>(
    std::uint64_t max_block_size);
#ifdef SCALUQ_USE_DEVICE
template void Circuit<internal::Prec>::optimize<ExecutionSpace::Default>(
    std::uint64_t max_block_size);
#endif  // SCALUQ_USE_DEVICE

namespace {
template <Precision Prec>
struct NoiseGate {
    std::span<const double> probabilities;
    std::vector<typename Circuit<Prec>::GateWithKey> alternatives;
};

template <Precision Prec>
NoiseGate<Prec> get_noise_gate(const typename Circuit<Prec>::GateWithKey& item) {
    if (const auto* gate = std::get_if<0>(&item)) {
        if (gate->gate_type() != GateType::Probabilistic) return {};
        const auto noise = ProbabilisticGate<Prec>(*gate);
        return {noise->distribution(), {noise->gate_list().begin(), noise->gate_list().end()}};
    }

    const auto& [gate, key] = std::get<1>(item);
    if (gate.param_gate_type() != ParamGateType::ParamProbabilistic) return {};
    const auto noise = ParamProbabilisticGate<Prec>(gate);
    NoiseGate<Prec> result{noise->distribution(), {}};
    result.alternatives.reserve(noise->gate_list().size());
    for (const auto& alternative : noise->gate_list()) {
        if (const auto* fixed_gate = std::get_if<0>(&alternative))
            result.alternatives.emplace_back(*fixed_gate);
        else
            result.alternatives.emplace_back(std::pair{std::get<1>(alternative), key});
    }
    return result;
}

struct NoiseBranch {
    std::uint64_t parent_index;
    std::uint64_t count;
};

struct NoiseSplit {
    std::vector<std::vector<NoiseBranch>> alternatives;
    std::size_t branch_count = 0;
};

template <class RandomEngineForRow>
NoiseSplit sample_noise_split(std::span<const double> probabilities,
                              std::span<const std::uint64_t> shot_counts,
                              const RandomEngineForRow& random_engine_for_row) {
    std::discrete_distribution<std::uint64_t> distribution(probabilities.begin(),
                                                           probabilities.end());
    NoiseSplit split{std::vector<std::vector<NoiseBranch>>(probabilities.size())};
    std::vector<std::uint64_t> counts(probabilities.size());
    for (std::uint64_t parent = 0; parent < shot_counts.size(); ++parent) {
        auto& random_engine = random_engine_for_row(parent);
        std::ranges::fill(counts, 0);
        for (std::uint64_t shot = 0; shot < shot_counts[parent]; ++shot)
            ++counts[distribution(random_engine)];
        for (std::size_t gate = 0; gate < counts.size(); ++gate) {
            if (counts[gate] == 0) continue;
            split.alternatives[gate].push_back({parent, counts[gate]});
            ++split.branch_count;
        }
    }
    return split;
}

template <class RandomEngineForRow>
void preserve_deterministic_draws(std::span<const std::uint64_t> shot_counts,
                                  const RandomEngineForRow& random_engine_for_row) {
    // Some standard libraries consume a random draw even with one possible result.
    constexpr std::array probabilities{1.};
    std::discrete_distribution<std::uint64_t> distribution(probabilities.begin(),
                                                           probabilities.end());
    for (std::uint64_t row = 0; row < shot_counts.size(); ++row) {
        auto& random_engine = random_engine_for_row(row);
        for (std::uint64_t shot = 0; shot < shot_counts[row]; ++shot)
            (void)distribution(random_engine);
    }
}

template <Precision Prec, ExecutionSpace Space>
void apply_noise_gate(const typename Circuit<Prec>::GateWithKey& item,
                      std::span<StateVector<Prec, Space>> states,
                      const std::map<std::string, double>& parameters) {
    if (const auto* fixed_gate = std::get_if<0>(&item)) {
        for (auto& state : states) (*fixed_gate)->update_quantum_state(state);
    } else {
        const auto& [gate, key] = std::get<1>(item);
        const auto parameter = parameters.at(key);
        for (auto& state : states) gate->update_quantum_state(state, parameter);
    }
}

template <Precision Prec, ExecutionSpace Space>
void apply_noise_gate(const typename Circuit<Prec>::GateWithKey& item,
                      StateVectorBatched<Prec, Space>& states,
                      std::span<const std::uint64_t> input_indices,
                      std::uint64_t input_size,
                      const std::map<std::string, std::vector<double>>& parameters) {
    if (const auto* fixed_gate = std::get_if<0>(&item)) {
        if (!input_indices.empty()) (*fixed_gate)->update_quantum_state(states);
    } else {
        const auto& [gate, key] = std::get<1>(item);
        const auto& input_parameters = parameters.at(key);
        if (input_parameters.size() != input_size)
            throw std::invalid_argument("Circuit::simulate_noise: parameter size mismatch.");
        if (input_indices.empty()) return;
        std::vector<double> branch_parameters;
        branch_parameters.reserve(input_indices.size());
        for (const auto index : input_indices) branch_parameters.push_back(input_parameters[index]);
        gate->update_quantum_state(states, branch_parameters);
    }
}

template <Precision Prec, ExecutionSpace Space>
auto gather_noise_branches(const typename Circuit<Prec>::template BatchedNoiseResult<Space>& source,
                           const NoiseSplit& split) ->
    typename Circuit<Prec>::template BatchedNoiseResult<Space> {
    using Batch = StateVectorBatched<Prec, Space>;
    using ExecSpace = internal::SpaceType<Space>;
    using GatherPolicy =
        Kokkos::MDRangePolicy<ExecSpace,
                              Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
    typename Circuit<Prec>::template BatchedNoiseResult<Space> next{
        Batch::uninitialized_state(split.branch_count, source.states.n_qubits()), {}, {}};
    next.input_indices.reserve(split.branch_count);
    next.counts.reserve(split.branch_count);
    std::vector<std::uint64_t> parents;
    parents.reserve(split.branch_count);
    for (const auto& branches : split.alternatives) {
        for (const auto& branch : branches) {
            parents.push_back(branch.parent_index);
            next.input_indices.push_back(source.input_indices[branch.parent_index]);
            next.counts.push_back(branch.count);
        }
    }
    if (!parents.empty()) {
        const auto source_view = source.states._raw;
        const auto target_view = next.states._raw;
        const auto device_parents = internal::convert_vector_to_view<std::uint64_t, Space>(parents);
        Kokkos::parallel_for(
            "gather_noise_branches",
            GatherPolicy({0, 0}, {split.branch_count, next.states.dim()}),
            KOKKOS_LAMBDA(std::uint64_t branch, std::uint64_t i) {
                target_view(branch, i) = source_view(device_parents(branch), i);
            });
    }
    return next;
}
}  // namespace

template <Precision Prec>
template <ExecutionSpace Space>
std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> Circuit<Prec>::simulate_noise(
    const StateVector<Prec, Space>& initial_state,
    std::uint64_t sampling_count,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const {
    if (has_classical_instructions()) {
        throw std::runtime_error(
            "Circuit::simulate_noise: Measurement and classically controlled instructions are "
            "not supported.");
    }
    check_state_vector_n_qubits(initial_state.n_qubits());
    if (initial_state.dim() == 0)
        throw std::runtime_error("Circuit::simulate_noise: initial state must be initialized.");
    std::mt19937 random_engine(internal::resolve_seed(seed));
    const auto random_engine_for_row =
        [engine = std::ref(random_engine)](std::uint64_t) -> std::mt19937& { return engine.get(); };

    std::vector<StateVector<Prec, Space>> states;
    if (sampling_count != 0 || _gate_list.empty()) states.push_back(initial_state.copy());
    std::vector<std::uint64_t> shot_counts(states.size(), sampling_count);

    for (const auto& item : _gate_list) {
        const auto noise = get_noise_gate<Prec>(item);
        if (noise.alternatives.empty()) {
            preserve_deterministic_draws(shot_counts, random_engine_for_row);
            apply_noise_gate<Prec, Space>(item, states, parameters);
            continue;
        }

        const auto split =
            sample_noise_split(noise.probabilities, shot_counts, random_engine_for_row);
        std::vector<std::size_t> remaining_children(states.size());
        for (const auto& branches : split.alternatives)
            for (const auto& branch : branches) ++remaining_children[branch.parent_index];

        std::vector<StateVector<Prec, Space>> next_states;
        std::vector<std::uint64_t> next_counts;
        next_states.reserve(split.branch_count);
        next_counts.reserve(split.branch_count);
        for (std::size_t gate = 0; gate < split.alternatives.size(); ++gate) {
            const auto begin = next_states.size();
            for (const auto& branch : split.alternatives[gate]) {
                auto& parent = states[branch.parent_index];
                // The last child can take over the parent's allocation.
                if (--remaining_children[branch.parent_index] == 0)
                    next_states.push_back(parent);
                else
                    next_states.push_back(parent.copy());
                next_counts.push_back(branch.count);
            }
            apply_noise_gate<Prec, Space>(
                noise.alternatives[gate], std::span(next_states).subspan(begin), parameters);
        }
        states = std::move(next_states);
        shot_counts = std::move(next_counts);
    }

    std::vector<std::pair<StateVector<Prec, Space>, std::int64_t>> result;
    result.reserve(states.size());
    for (std::size_t i = 0; i < states.size(); ++i) result.emplace_back(states[i], shot_counts[i]);
    return result;
}

template <Precision Prec>
template <ExecutionSpace Space>
auto Circuit<Prec>::simulate_noise(const StateVectorBatched<Prec, Space>& initial_states,
                                   std::uint64_t sampling_count,
                                   const std::map<std::string, std::vector<double>>& parameters,
                                   std::optional<std::uint64_t> seed) const
    -> BatchedNoiseResult<Space> {
    if (has_classical_instructions()) {
        throw std::runtime_error(
            "Circuit::simulate_noise: Measurement and classically controlled instructions are "
            "not supported.");
    }
    check_state_vector_n_qubits(initial_states.n_qubits());
    using Batch = StateVectorBatched<Prec, Space>;
    const auto input_size = initial_states.batch_size();
    const auto n_qubits = initial_states.n_qubits();
    const auto active_size = sampling_count == 0 ? 0 : input_size;
    BatchedNoiseResult<Space> result{
        active_size == 0 ? Batch::uninitialized_state(0, n_qubits) : initial_states.copy(),
        std::vector<std::uint64_t>(active_size),
        std::vector<std::uint64_t>(active_size, sampling_count)};
    std::iota(result.input_indices.begin(), result.input_indices.end(), 0ULL);

    const auto first_seed = internal::resolve_seed(seed);
    std::vector<std::mt19937> random_engines;
    random_engines.reserve(active_size);
    for (std::uint64_t row = 0; row < active_size; ++row)
        random_engines.emplace_back(first_seed + row);
    const auto random_engine_for_row = [&](std::uint64_t row) -> auto& {
        return random_engines[result.input_indices[row]];
    };

    for (const auto& item : _gate_list) {
        const auto noise = get_noise_gate<Prec>(item);
        if (noise.alternatives.empty()) {
            preserve_deterministic_draws(result.counts, random_engine_for_row);
            apply_noise_gate<Prec, Space>(
                item, result.states, result.input_indices, input_size, parameters);
            continue;
        }

        const auto split =
            sample_noise_split(noise.probabilities, result.counts, random_engine_for_row);
        auto next = gather_noise_branches<Prec, Space>(result, split);
        std::size_t begin = 0;
        for (std::size_t gate = 0; gate < split.alternatives.size(); ++gate) {
            const auto size = split.alternatives[gate].size();
            if (size == 0) {
                // Preserve parameter validation for alternatives with no selected branches.
                apply_noise_gate<Prec, Space>(
                    noise.alternatives[gate], next.states, {}, input_size, parameters);
            } else {
                auto branches = next.states.view_batch_range(begin, begin + size);
                const auto destination = branches._raw;
                apply_noise_gate<Prec, Space>(noise.alternatives[gate],
                                              branches,
                                              std::span(next.input_indices).subspan(begin, size),
                                              input_size,
                                              parameters);
                // Some gates, such as SparseMatrix, replace their state allocation.
                if (branches._raw.data() != destination.data())
                    Kokkos::deep_copy(destination, branches._raw);
            }
            begin += size;
        }
        result = std::move(next);
    }
    return result;
}

template std::vector<std::pair<StateVector<internal::Prec, ExecutionSpace::Host>, std::int64_t>>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::Host>(
    const StateVector<internal::Prec, ExecutionSpace::Host>& initial_state,
    std::uint64_t sampling_count,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template Circuit<internal::Prec>::BatchedNoiseResult<ExecutionSpace::Host>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::Host>(
    const StateVectorBatched<internal::Prec, ExecutionSpace::Host>& initial_states,
    std::uint64_t sampling_count,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;

template std::vector<
    std::pair<StateVector<internal::Prec, ExecutionSpace::HostSerial>, std::int64_t>>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::HostSerial>(
    const StateVector<internal::Prec, ExecutionSpace::HostSerial>& initial_state,
    std::uint64_t sampling_count,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template Circuit<internal::Prec>::BatchedNoiseResult<ExecutionSpace::HostSerial>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::HostSerial>(
    const StateVectorBatched<internal::Prec, ExecutionSpace::HostSerial>& initial_states,
    std::uint64_t sampling_count,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
#ifdef SCALUQ_USE_DEVICE
template std::vector<std::pair<StateVector<internal::Prec, ExecutionSpace::Default>, std::int64_t>>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::Default>(
    const StateVector<internal::Prec, ExecutionSpace::Default>& initial_state,
    std::uint64_t sampling_count,
    const std::map<std::string, double>& parameters,
    std::optional<std::uint64_t> seed) const;
template Circuit<internal::Prec>::BatchedNoiseResult<ExecutionSpace::Default>
Circuit<internal::Prec>::simulate_noise<ExecutionSpace::Default>(
    const StateVectorBatched<internal::Prec, ExecutionSpace::Default>& initial_states,
    std::uint64_t sampling_count,
    const std::map<std::string, std::vector<double>>& parameters,
    std::optional<std::uint64_t> seed) const;
#endif  // SCALUQ_USE_DEVICE

template <Precision Prec>
std::uint64_t Circuit<Prec>::required_n_qubits() const {
    std::uint64_t operand_mask = 0;
    for (const auto& gate : _gate_list) {
        operand_mask |= gate.index() == 0 ? std::get<0>(gate)->operand_qubit_mask()
                                          : std::get<1>(gate).first->operand_qubit_mask();
    }
    return std::bit_width(operand_mask);
}

template <Precision Prec>
void Circuit<Prec>::check_state_vector_n_qubits(std::uint64_t n_qubits) const {
    if (required_n_qubits() > n_qubits) {
        throw std::runtime_error(
            "Circuit::update_quantum_state: state does not have enough qubits for this circuit.");
    }
}

template <Precision Prec>
bool Circuit<Prec>::has_classical_instructions() const {
    if (std::ranges::any_of(_classical_conditions,
                            [](const auto& condition) { return condition.has_value(); })) {
        return true;
    }
    return std::ranges::any_of(_gate_list, [](const GateWithKey& gate) {
        return gate.index() == 0 && std::get<0>(gate).gate_type() == GateType::Measurement;
    });
}

template <Precision Prec>
template <ExecutionSpace Space>
// Low-level implementation for expectation gradient that assumes the forward state and
// observable-applied bistate are already prepared, and accumulates gradient by
// traversing the circuit in reverse with backpropagation.
std::unordered_map<std::string, double> Circuit<Prec>::compute_expectation_gradient_backprop(
    StateVector<Prec, Space>& state,
    StateVector<Prec, Space>& bistate,
    const std::map<std::string, double>& parameters) {
    check_gradient_circuit(*this, parameters);
    if (state.n_qubits() != bistate.n_qubits()) {
        throw std::invalid_argument("Expectation gradient: state dimensions must match.");
    }
    check_state_vector_n_qubits(state.n_qubits());
    const std::uint64_t n_gates = this->n_gates();

    std::unordered_map<std::string, double> gradient;
    const auto& key_set = this->key_set();
    gradient.reserve(key_set.size());
    for (const auto& key : key_set) {
        gradient.emplace(key, 0.0);
    }

    StateVector<Prec, Space> Astate(state.n_qubits());

    for (std::int64_t i = static_cast<std::int64_t>(n_gates) - 1; i >= 0; i--) {
        const auto& cur_gate = this->get_gate_at(static_cast<std::uint64_t>(i));

        if (cur_gate.index() == 1) {
            const auto& [pgate, key] = std::get<1>(cur_gate);

            std::uint64_t x_mask = 0, z_mask = 0;
            double pauli_coef = 1.;
            switch (pgate.param_gate_type()) {
                case ParamGateType::ParamRX:
                    x_mask = pgate->target_qubit_mask();
                    break;
                case ParamGateType::ParamRY:
                    x_mask = z_mask = pgate->target_qubit_mask();
                    break;
                case ParamGateType::ParamRZ:
                    z_mask = pgate->target_qubit_mask();
                    break;
                case ParamGateType::ParamPauliRotation: {
                    internal::ParamGatePtr<internal::ParamPauliRotationGateImpl<Prec>> ppr(pgate);
                    const auto pauli = ppr->pauli();
                    const auto coef = pauli.coef();
                    if (!std::isfinite(coef.real()) || coef.imag() != 0.) {
                        throw std::invalid_argument(
                            "Expectation gradient requires a finite real Pauli coefficient.");
                    }
                    std::tie(x_mask, z_mask) = pauli.get_XZ_mask_representation();
                    pauli_coef = coef.real();
                    break;
                }
                default:
                    throw std::invalid_argument(
                        "Expectation gradient does not support this parameterized gate.");
            }
            const double scale = pgate->param_coef() * pauli_coef;
            if (!std::isfinite(scale)) {
                throw std::invalid_argument(
                    "Expectation gradient requires finite gate coefficients.");
            }
            if (scale != 0.) {
                Astate = state.copy();
                internal::apply_pauli(
                    0ULL, 0ULL, x_mask, z_mask, internal::Complex<Prec>(1.), Astate);
                const auto control_mask = pgate->control_qubit_mask();
                const auto control_values = pgate->control_value_mask();
                if (control_mask) {
                    const auto amplitudes = Astate._raw;
                    Kokkos::parallel_for(
                        "gradient_control_projection",
                        Kokkos::RangePolicy<internal::SpaceType<Space>>(0, Astate.dim()),
                        KOKKOS_LAMBDA(std::uint64_t basis) {
                            if ((basis & control_mask) != control_values) amplitudes(basis) = 0.;
                        });
                }
                // d<E>/dt = scale * Im(<bistate| control_projector * Pauli |state>).
                const auto ip = internal::inner_product<Prec, Space>(bistate._raw, Astate._raw);
                gradient[key] += scale * static_cast<double>(ip.imag());
            }
        }

        if (cur_gate.index() == 0) {
            const auto& g = std::get<0>(cur_gate);

            auto inv = g->get_inverse();
            inv->update_quantum_state(bistate);
            inv->update_quantum_state(state);
        } else {
            const auto& [pgate, key] = std::get<1>(cur_gate);

            auto inv = pgate->get_inverse();
            const auto it = parameters.find(key);
            if (it == parameters.end()) {
                throw std::runtime_error(
                    "Circuit::compute_expectation_gradient_backprop: parameter named " + key +
                    " is not given.");
            }
            const auto param = it->second;

            inv->update_quantum_state(bistate, param);
            inv->update_quantum_state(state, param);
        }
    }
    return gradient;
}
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient_backprop(
    StateVector<internal::Prec, ExecutionSpace::Host>& state,
    StateVector<internal::Prec, ExecutionSpace::Host>& bistate,
    const std::map<std::string, double>& parameters);
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient_backprop(
    StateVector<internal::Prec, ExecutionSpace::HostSerial>& state,
    StateVector<internal::Prec, ExecutionSpace::HostSerial>& bistate,
    const std::map<std::string, double>& parameters);
#ifdef SCALUQ_USE_DEVICE
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient_backprop(
    StateVector<internal::Prec, ExecutionSpace::Default>& state,
    StateVector<internal::Prec, ExecutionSpace::Default>& bistate,
    const std::map<std::string, double>& parameters);
#endif  // SCALUQ_USE_DEVICE

template <Precision Prec>
template <ExecutionSpace Space>
std::unordered_map<std::string, double> Circuit<Prec>::compute_expectation_gradient(
    const Operator<Prec, Space>& obs, const std::map<std::string, double>& parameters) {
    if (!obs.is_hermitian()) {
        throw std::runtime_error(
            "Circuit::compute_expectation_gradient: observable must be Hermitian.");
    }
    StateVector<Prec, Space> state(required_n_qubits());
    this->update_quantum_state(state, parameters);

    StateVector<Prec, Space> bistate = state.copy();
    obs.apply_to_state(bistate);
    return compute_expectation_gradient_backprop(state, bistate, parameters);
}
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient<ExecutionSpace::Host>(
    const Operator<internal::Prec, ExecutionSpace::Host>& obs,
    const std::map<std::string, double>& parameters);
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient<ExecutionSpace::HostSerial>(
    const Operator<internal::Prec, ExecutionSpace::HostSerial>& obs,
    const std::map<std::string, double>& parameters);
#ifdef SCALUQ_USE_DEVICE
template std::unordered_map<std::string, double>
Circuit<internal::Prec>::compute_expectation_gradient<ExecutionSpace::Default>(
    const Operator<internal::Prec, ExecutionSpace::Default>& obs,
    const std::map<std::string, double>& parameters);
#endif  // SCALUQ_USE_DEVICE

template class Circuit<internal::Prec>;
}  // namespace scaluq
