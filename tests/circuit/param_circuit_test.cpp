#include <gtest/gtest.h>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class ParamCircuitTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(ParamCircuitTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
void param_circuit_test() {
    std::uint64_t n_qubits = 5;
    Random random;
    for ([[maybe_unused]] std::uint64_t repeat : std::views::iota(0, 10)) {
        Circuit<Prec, Space> circuit(n_qubits);
        Circuit<Prec, Space> pcircuit(n_qubits);
        constexpr std::uint64_t nparams = 5;
        std::array<std::string, nparams> pkeys = {};
        for (std::uint64_t pidx : std::views::iota(std::uint64_t{0}, nparams))
            pkeys[pidx] = "p" + std::to_string(pidx);
        std::array<double, nparams> params = {};
        for (std::uint64_t pidx : std::views::iota(std::uint64_t{0}, nparams))
            params[pidx] = random.uniform() * std::numbers::pi * 2;
        std::map<std::string, double> pmap;
        for (std::uint64_t pidx : std::views::iota(std::uint64_t{0}, nparams))
            pmap[pkeys[pidx]] = params[pidx];
        auto add_random_rotation_or_cnot = [&] {
            if (random.int32() & 1) {
                std::uint64_t param_gate_kind = random.int32() % 4;
                double coef = random.uniform() * 2 - 1;
                std::uint64_t pidx = random.int32() % nparams;
                if (param_gate_kind == 0) {
                    std::uint64_t target = random.int32() % n_qubits;
                    circuit.add_gate(gate::RX<Prec, Space>(target, coef * params[pidx]));
                    pcircuit.add_param_gate(gate::ParamRX<Prec, Space>(target, coef), pkeys[pidx]);
                } else if (param_gate_kind == 1) {
                    std::uint64_t target = random.int32() % n_qubits;
                    circuit.add_gate(gate::RY<Prec, Space>(target, coef * params[pidx]));
                    pcircuit.add_param_gate(gate::ParamRY<Prec, Space>(target, coef), pkeys[pidx]);
                } else if (param_gate_kind == 2) {
                    std::uint64_t target = random.int32() % n_qubits;
                    circuit.add_gate(gate::RZ<Prec, Space>(target, coef * params[pidx]));
                    pcircuit.add_param_gate(gate::ParamRZ<Prec, Space>(target, coef), pkeys[pidx]);
                } else {
                    std::vector<std::uint64_t> target_vec, pauli_id_vec;
                    for (std::uint64_t target = 0; target < n_qubits; target++) {
                        target_vec.emplace_back(target);
                        pauli_id_vec.emplace_back(random.int64() % 4);
                    }
                    PauliOperator<Prec, Space> pauli(target_vec, pauli_id_vec, 1.0);
                    circuit.add_gate(gate::PauliRotation<Prec, Space>(pauli, coef * params[pidx]));
                    pcircuit.add_param_gate(gate::ParamPauliRotation<Prec, Space>(pauli, coef),
                                            pkeys[pidx]);
                }
            } else {
                std::uint64_t control = random.int32() % n_qubits;
                std::uint64_t target = random.int32() % (n_qubits - 1);
                if (target == control) target = n_qubits - 1;
                circuit.add_gate(gate::CX<Prec, Space>(control, target));
                pcircuit.add_gate(gate::CX<Prec, Space>(control, target));
            }
        };
        for ([[maybe_unused]] std::uint64_t _ : std::views::iota(0ULL, 20ULL)) {
            add_random_rotation_or_cnot();
        }
        StateVector state = StateVector<Prec, Space>::Haar_random_state(n_qubits);
        StateVector state_cp = state.copy();
        auto amplitudes_base = state.get_amplitudes();
        circuit.update_quantum_state(state);
        pcircuit.update_quantum_state(state_cp, pmap);
        auto amplitudes = state.get_amplitudes();
        auto amplitudes_cp = state_cp.get_amplitudes();
        for (std::uint64_t idx : std::views::iota(std::uint64_t{0}, 1ULL << n_qubits)) {
            check_near<Prec>(amplitudes[idx], amplitudes_cp[idx]);
        }
        auto icircuit = circuit.get_inverse();
        auto ipcircuit = pcircuit.get_inverse();
        icircuit.update_quantum_state(state);
        ipcircuit.update_quantum_state(state_cp, pmap);
        amplitudes = state.get_amplitudes();
        amplitudes_cp = state_cp.get_amplitudes();
        for (std::uint64_t idx : std::views::iota(std::uint64_t{0}, 1ULL << n_qubits)) {
            check_near<Prec>(amplitudes[idx], amplitudes_base[idx]);
            check_near<Prec>(amplitudes_cp[idx], amplitudes_base[idx]);
        }
    }
}

TYPED_TEST(ParamCircuitTest, ApplyParamCircuit) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    param_circuit_test<Prec, Space>();
}

TYPED_TEST(ParamCircuitTest, InsufficientParameterGiven) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    {
        Circuit<Prec, Space> circuit(1);
        circuit.add_param_gate(gate::ParamRX<Prec, Space>(0), "0");
        circuit.add_param_gate(gate::ParamRX<Prec, Space>(0), "1");
        circuit.add_param_gate(gate::ParamRX<Prec, Space>(0), "0");
        StateVector<Prec, Space> state(1);
        ASSERT_NO_THROW(circuit.update_quantum_state(state, {{"0", 0}, {"1", 0}}));
        ASSERT_NO_THROW(circuit.update_quantum_state(state, {{"0", 0}, {"1", 0}, {"2", 0}}));
        ASSERT_THROW(circuit.update_quantum_state(state), std::runtime_error);
        ASSERT_THROW(circuit.update_quantum_state(state, {}), std::runtime_error);
        ASSERT_THROW(circuit.update_quantum_state(state, {{"0", 0}}), std::runtime_error);
        ASSERT_THROW(circuit.update_quantum_state(state, {{"0", 0}, {"2", 0}}), std::runtime_error);
    }
}
