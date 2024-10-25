#include <gtest/gtest.h>

#include <gate/gate_factory.hpp>
#include <gate/merge_gate.hpp>
#include <numbers>
#include <ranges>
#include <state/state_vector.hpp>
#include <types.hpp>
#include <util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

TEST(GateTest, MergeGate) {
    std::vector<Gate> gates;
    Random random;
    std::uint64_t n = 4;
    auto none_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::vector<std::uint64_t> controls(shuffled.begin(), shuffled.begin() + nc);
            gates.push_back(fac(random.uniform() * std::numbers::pi * 2, controls));
        }
    };
    auto single_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            gates.push_back(fac(target, controls));
        }
    };
    auto single_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            gates.push_back(fac(target, random.uniform() * std::numbers::pi * 2, controls));
        }
    };
    auto single_target_rotation2 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls));
        }
    };
    auto single_target_rotation3 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls));
        }
    };
    auto double_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target0 = shuffled[0];
            std::uint64_t target1 = shuffled[1];
            std::vector<std::uint64_t> controls(shuffled.begin() + 2, shuffled.begin() + 2 + nc);
            gates.push_back(fac(target0, target1, controls));
        }
    };
    auto dense_matrix = [&](auto fac) {
        for (auto nt : {0, 1, 2, 3}) {
            for (auto nc : {0, 1, 2}) {
                if (nt + nc > static_cast<int>(n)) continue;
                std::vector<std::uint64_t> shuffled = random.permutation(n);
                std::vector<uint64_t> targets(shuffled.begin(), shuffled.begin() + nt);
                std::vector<std::uint64_t> controls(shuffled.begin() + nt,
                                                    shuffled.begin() + nt + nc);
                internal::ComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        mat(i, j) = StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                    }
                }
                gates.push_back(fac(targets, mat, controls, false));
            }
        }
    };
    auto sparse_matrix = [&](auto fac) {
        for (auto nt : {0, 1, 2, 3}) {
            for (auto nc : {0, 1, 2}) {
                if (nt + nc > static_cast<int>(n)) continue;
                std::vector<std::uint64_t> shuffled = random.permutation(n);
                std::vector<uint64_t> targets(shuffled.begin(), shuffled.begin() + nt);
                std::vector<std::uint64_t> controls(shuffled.begin() + nt,
                                                    shuffled.begin() + nt + nc);
                internal::SparseComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        if (random.uniform() < .5) {
                            mat.insert(i, j) =
                                StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                        }
                    }
                }
                gates.push_back(fac(targets, mat, controls));
            }
        }
    };
    gates.push_back(gate::I());
    none_target_rotation(gate::GlobalPhase);
    single_target(gate::X);
    single_target(gate::Y);
    single_target(gate::Z);
    single_target(gate::H);
    single_target(gate::S);
    single_target(gate::Sdag);
    single_target(gate::T);
    single_target(gate::Tdag);
    single_target(gate::SqrtX);
    single_target(gate::SqrtXdag);
    single_target(gate::SqrtY);
    single_target(gate::SqrtYdag);
    single_target(gate::P0);
    single_target(gate::P1);
    single_target_rotation(gate::RX);
    single_target_rotation(gate::RY);
    single_target_rotation(gate::RZ);
    single_target_rotation(gate::U1);
    single_target_rotation2(gate::U2);
    single_target_rotation3(gate::U3);
    double_target(gate::Swap);
    dense_matrix(gate::DenseMatrix);
    sparse_matrix(gate::SparseMatrix);
    gates.push_back(gate::Pauli(PauliOperator("X 0 Y 2", random.uniform())));
    gates.push_back(gate::Pauli(PauliOperator("Z 0", random.uniform())));
    gates.push_back(gate::Pauli(PauliOperator("Z 3", random.uniform()), {1}));
    gates.push_back(gate::Pauli(PauliOperator("Z 1", random.uniform()), {0, 3}));
    gates.push_back(gate::PauliRotation(PauliOperator("X 0 Y 2", random.uniform()),
                                        random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation(PauliOperator("Z 0", random.uniform()),
                                        random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation(
        PauliOperator("Z 3", random.uniform()), random.uniform() * std::numbers::pi * 2, {1}));
    gates.push_back(gate::PauliRotation(
        PauliOperator("Z 1", random.uniform()), random.uniform() * std::numbers::pi * 2, {0, 3}));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            std::cerr << "====" << std::endl;
            std::cerr << g1 << std::endl;
            std::cerr << g2 << std::endl;
            std::cerr << g2->get_matrix() << std::endl;
            std::cerr << "====" << std::endl;
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
