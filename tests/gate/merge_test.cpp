#include <gtest/gtest.h>

#include <gate/gate_factory.hpp>
#include <gate/merge_gate.hpp>
#include <numbers>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

TEST(GateTest, MergeGate) {
    std::vector<Gate> gates;
    Random random;
    for (std::uint64_t target = 0; target < 2; target++) {
        gates.push_back(gate::X(target));
        gates.push_back(gate::Y(target));
        gates.push_back(gate::Z(target));
        gates.push_back(gate::H(target));
        gates.push_back(gate::S(target));
        gates.push_back(gate::Sdag(target));
        gates.push_back(gate::T(target));
        gates.push_back(gate::Tdag(target));
        gates.push_back(gate::SqrtX(target));
        gates.push_back(gate::SqrtXdag(target));
        gates.push_back(gate::SqrtY(target));
        gates.push_back(gate::SqrtYdag(target));
        gates.push_back(gate::P0(target));
        gates.push_back(gate::P1(target));
        gates.push_back(gate::RX(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::RY(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::RZ(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U1(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U2(target,
                                 random.uniform() * std::numbers::pi * 2,
                                 random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U3(target,
                                 random.uniform() * std::numbers::pi * 2,
                                 random.uniform() * std::numbers::pi * 2,
                                 random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::CX(target, target ^ 1));
        gates.push_back(gate::CZ(target, target ^ 1));
        gates.push_back(gate::Swap(target, target ^ 1));
    }
    gates.push_back(gate::I());
    gates.push_back(gate::GlobalPhase(random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::Pauli(PauliOperator("X 0 Y 1", random.uniform())));
    gates.push_back(gate::Pauli(PauliOperator("Z 0", random.uniform())));
    gates.push_back(gate::Pauli(PauliOperator("Z 1", random.uniform())));
    gates.push_back(gate::PauliRotation(PauliOperator("X 0 Y 1", random.uniform()),
                                        random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation(PauliOperator("Z 0", random.uniform()),
                                        random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation(PauliOperator("Z 1", random.uniform()),
                                        random.uniform() * std::numbers::pi * 2));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            std::uint64_t n = 2;
            auto state1 = StateVector::Haar_random_state(n);
            auto state2 = state1.copy();
            auto [mg, phase] = merge_gate(g1, g2);
            g1->update_quantum_state(state1);
            g2->update_quantum_state(state1);
            mg->update_quantum_state(state2);
            state2.multiply_coef(Kokkos::polar(1., phase));
            ASSERT_TRUE(same_state(state1, state2));
        }
    }
}
