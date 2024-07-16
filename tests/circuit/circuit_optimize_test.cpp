#include <gtest/gtest.h>

#include <circuit/circuit.hpp>
#include <gate/gate_factory.hpp>
#include <gate/param_gate_factory.hpp>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

TEST(CircuitTest, Optimize1) {
    UINT n = 5;
    Random random;
    Circuit circuit(n);
    UINT param_id = 0;
    for ([[maybe_unused]] UINT _ : std::views::iota(0, 1000)) {
        UINT kind = random.int32();
        UINT q0 = random.int32() % n;
        UINT q1 = random.int32() % (n - 1);
        if (q1 == q0) q1 = n - 1;
        double r0 = random.uniform() * PI() * 2;
        double r1 = random.uniform() * PI() * 2;
        double r2 = random.uniform() * PI() * 2;
        auto random_pauli = [&] {
            std::vector<UINT> target_list;
            std::vector<UINT> pauli_id_list;
            for (UINT q : std::views::iota(0ULL, n)) {
                if (random.int32() & 1) {
                    target_list.push_back(q);
                    pauli_id_list.push_back(random.int32() & 3);
                }
            }
            return PauliOperator(target_list, pauli_id_list, random.uniform());
        };
        auto gen_param = [&] { return std::to_string(param_id++); };
        switch (kind) {
            case 0:
                circuit.add_gate(gate::I());
                break;
            case 1:
                circuit.add_gate(gate::GlobalPhase(random.uniform() * PI() * 2));
                break;
            case 2:
                circuit.add_gate(gate::X(q0));
                break;
            case 3:
                circuit.add_gate(gate::Y(q0));
                break;
            case 4:
                circuit.add_gate(gate::Z(q0));
                break;
            case 5:
                circuit.add_gate(gate::H(q0));
                break;
            case 6:
                circuit.add_gate(gate::S(q0));
                break;
            case 7:
                circuit.add_gate(gate::Sdag(q0));
                break;
            case 8:
                circuit.add_gate(gate::T(q0));
                break;
            case 9:
                circuit.add_gate(gate::Tdag(q0));
                break;
            case 10:
                circuit.add_gate(gate::SqrtX(q0));
                break;
            case 11:
                circuit.add_gate(gate::SqrtXdag(q0));
                break;
            case 12:
                circuit.add_gate(gate::SqrtY(q0));
                break;
            case 13:
                circuit.add_gate(gate::SqrtYdag(q0));
                break;
            case 14:
                circuit.add_gate(gate::P0(q0));
                break;
            case 15:
                circuit.add_gate(gate::RX(q0, r0));
                break;
            case 16:
                circuit.add_gate(gate::RY(q0, r0));
                break;
            case 17:
                circuit.add_gate(gate::RZ(q0, r0));
                break;
            case 18:
                circuit.add_gate(gate::U1(q0, r0));
                break;
            case 19:
                circuit.add_gate(gate::U2(q0, r0, r1));
                break;
            case 20:
                circuit.add_gate(gate::U3(q0, r0, r1, r2));
                break;
            case 21:
                circuit.add_gate(gate::CX(q0, q1));
                break;
            case 22:
                circuit.add_gate(gate::CZ(q0, q1));
                break;
            case 23:
                circuit.add_gate(gate::Swap(q0, q1));
                break;
            case 24: {
                UINT block_size = random.int32() % (n / 2) + 1;
                UINT q0 = random.int32() % (n - block_size * 2 + 1);
                UINT q1 = random.int32() % (n - block_size * 2 + 1);
                if (q0 > q1) std::swap(q0, q1);
                q1 += block_size;
                circuit.add_gate(gate::FusedSwap(q0, q1, block_size));
            } break;
            case 25:
                circuit.add_gate(gate::OneQubitMatrix(
                    q0,
                    {std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())},
                     std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())}}));
                break;
            case 26:
                circuit.add_gate(gate::TwoQubitMatrix(
                    q0,
                    q1,
                    {std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())},
                     std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())},
                     std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())},
                     std::array{Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform()),
                                Complex(random.uniform(), random.uniform())}}));
                break;
            case 27:
                circuit.add_gate(gate::Pauli(random_pauli()));
                break;
            case 28:
                circuit.add_gate(gate::PauliRotation(random_pauli(), r0));
                break;
            case 29:
                circuit.add_param_gate(gate::PRX(q0, random.uniform()), gen_param());
                break;
            case 30:
                circuit.add_param_gate(gate::PRY(q0, random.uniform()), gen_param());
                break;
            case 31:
                circuit.add_param_gate(gate::PRZ(q0, random.uniform()), gen_param());
                break;
            case 32:
                circuit.add_param_gate(gate::PPauliRotation(random_pauli(), random.uniform()),
                                       gen_param());
                break;
        }
    }
    std::map<std::string, double> params;
    for (UINT pid : std::views::iota(0ULL, param_id)) {
        params[std::to_string(pid)] = random.uniform() * PI() * 2;
    }
    auto state0 = StateVector::Haar_random_state(n);
    auto state1 = state0.copy();
    circuit.update_quantum_state(state0, params);
    UINT ngates = circuit.gate_count();
    circuit.optimize();
    circuit.update_quantum_state(state1, params);
    ASSERT_LT(circuit.gate_count(), ngates);
    ASSERT_TRUE(same_state(state0, state1));
}
