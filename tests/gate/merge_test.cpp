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
    std::vector<Gate<double>> gates;
    Random random;
    for (std::uint64_t target = 0; target < 2; target++) {
        gates.push_back(gate::X<double>(target));
        gates.push_back(gate::Y<double>(target));
        gates.push_back(gate::Z<double>(target));
        gates.push_back(gate::H<double>(target));
        gates.push_back(gate::S<double>(target));
        gates.push_back(gate::Sdag<double>(target));
        gates.push_back(gate::T<double>(target));
        gates.push_back(gate::Tdag<double>(target));
        gates.push_back(gate::SqrtX<double>(target));
        gates.push_back(gate::SqrtXdag<double>(target));
        gates.push_back(gate::SqrtY<double>(target));
        gates.push_back(gate::SqrtYdag<double>(target));
        gates.push_back(gate::P0<double>(target));
        gates.push_back(gate::P1<double>(target));
        gates.push_back(gate::RX<double>(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::RY<double>(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::RZ<double>(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U1<double>(target, random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U2<double>(target,
                                         random.uniform() * std::numbers::pi * 2,
                                         random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::U3<double>(target,
                                         random.uniform() * std::numbers::pi * 2,
                                         random.uniform() * std::numbers::pi * 2,
                                         random.uniform() * std::numbers::pi * 2));
        gates.push_back(gate::OneTargetMatrix<double>(
            target,
            {std::array{Complex<double>(random.uniform(), random.uniform()),
                        Complex<double>(random.uniform(), random.uniform())},
             std::array{Complex<double>(random.uniform(), random.uniform()),
                        Complex<double>(random.uniform(), random.uniform())}}));
        gates.push_back(gate::CX<double>(target, target ^ 1));
        gates.push_back(gate::CZ<double>(target, target ^ 1));
        gates.push_back(gate::Swap<double>(target, target ^ 1));
    }
    gates.push_back(gate::I<double>());
    gates.push_back(gate::GlobalPhase<double>(random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::TwoTargetMatrix<double>(
        0,
        1,
        {std::array{Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform())},
         std::array{Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform())},
         std::array{Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform())},
         std::array{Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform()),
                    Complex<double>(random.uniform(), random.uniform())}}));
    gates.push_back(gate::Pauli<double>(PauliOperator<double>("X 0 Y 1", random.uniform())));
    gates.push_back(gate::Pauli<double>(PauliOperator<double>("Z 0", random.uniform())));
    gates.push_back(gate::Pauli<double>(PauliOperator<double>("Z 1", random.uniform())));
    gates.push_back(gate::PauliRotation<double>(PauliOperator<double>("X 0 Y 1", random.uniform()),
                                                random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation<double>(PauliOperator<double>("Z 0", random.uniform()),
                                                random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation<double>(PauliOperator<double>("Z 1", random.uniform()),
                                                random.uniform() * std::numbers::pi * 2));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            std::uint64_t n = 2;
            auto state1 = StateVector<double>::Haar_random_state(n);
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
