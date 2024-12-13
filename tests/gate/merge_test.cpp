#include <gtest/gtest.h>

#include <numbers>
#include <ranges>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/merge_gate.hpp>
#include <scaluq/state/state_vector.hpp>
#include <scaluq/types.hpp>
#include <scaluq/util/random.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <std::floating_point Fp>
void merge_gate_test() {
    std::vector<Gate<Fp>> gates;
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
                internal::ComplexMatrix<Fp> mat(1 << nt, 1 << nt);
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
                internal::SparseComplexMatrix<Fp> mat(1 << nt, 1 << nt);
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
    gates.push_back(gate::I<Fp>());
    none_target_rotation(gate::GlobalPhase<Fp>);
    single_target(gate::X<Fp>);
    single_target(gate::Y<Fp>);
    single_target(gate::Z<Fp>);
    single_target(gate::H<Fp>);
    single_target(gate::S<Fp>);
    single_target(gate::Sdag<Fp>);
    single_target(gate::T<Fp>);
    single_target(gate::Tdag<Fp>);
    single_target(gate::SqrtX<Fp>);
    single_target(gate::SqrtXdag<Fp>);
    single_target(gate::SqrtY<Fp>);
    single_target(gate::SqrtYdag<Fp>);
    single_target(gate::P0<Fp>);
    single_target(gate::P1<Fp>);
    single_target_rotation(gate::RX<Fp>);
    single_target_rotation(gate::RY<Fp>);
    single_target_rotation(gate::RZ<Fp>);
    single_target_rotation(gate::U1<Fp>);
    single_target_rotation2(gate::U2<Fp>);
    single_target_rotation3(gate::U3<Fp>);
    double_target(gate::Swap<Fp>);
    dense_matrix(gate::DenseMatrix<Fp>);
    sparse_matrix(gate::SparseMatrix<Fp>);
    gates.push_back(gate::Pauli<Fp>(PauliOperator<Fp>("X 0 Y 2", random.uniform())));
    gates.push_back(gate::Pauli<Fp>(PauliOperator<Fp>("Z 0", random.uniform())));
    gates.push_back(gate::Pauli<Fp>(PauliOperator<Fp>("Z 3", random.uniform()), {1}));
    gates.push_back(gate::Pauli<Fp>(PauliOperator<Fp>("Z 1", random.uniform()), {0, 3}));
    gates.push_back(gate::PauliRotation<Fp>(PauliOperator<Fp>("X 0 Y 2", random.uniform()),
                                            random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation<Fp>(PauliOperator<Fp>("Z 0", random.uniform()),
                                            random.uniform() * std::numbers::pi * 2));
    gates.push_back(gate::PauliRotation<Fp>(
        PauliOperator<Fp>("Z 3", random.uniform()), random.uniform() * std::numbers::pi * 2, {1}));
    gates.push_back(gate::PauliRotation<Fp>(PauliOperator<Fp>("Z 1", random.uniform()),
                                            random.uniform() * std::numbers::pi * 2,
                                            {0, 3}));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            auto state1 = StateVector<Fp>::Haar_random_state(n);
            auto state2 = state1.copy();
            auto [mg, phase] = merge_gate(g1, g2);
            g1->update_quantum_state(state1);
            g2->update_quantum_state(state1);
            mg->update_quantum_state(state2);
            state2.multiply_coef(Kokkos::polar(static_cast<Fp>(1.), phase));
            ASSERT_TRUE(same_state(state1, state2));
        }
    }
}

TEST(GateTest, MergeGate) {
    merge_gate_test<double>();
    merge_gate_test<float>();
}
