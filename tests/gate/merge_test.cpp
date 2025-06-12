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

template <typename T>
class MergeGateTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(MergeGateTest, TestTypes, NameGenerator);

template <Precision Prec, ExecutionSpace Space>
void merge_gate_test() {
    std::vector<Gate<Prec, Space>> gates;
    Random random;
    std::uint64_t n = 4;

    auto make_control_values = [&](std::uint64_t length) {
        std::uint64_t control_value_mask = random.int64();
        std::vector<std::uint64_t> control_values(length);
        for (std::uint64_t i = 0; i < length; ++i) control_values[i] = control_value_mask >> i & 1;
        return control_values;
    };

    auto none_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::vector<std::uint64_t> controls(shuffled.begin(), shuffled.begin() + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(random.uniform() * std::numbers::pi * 2, controls, control_values));
        }
    };
    auto single_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target, controls, control_values));
        }
    };
    auto single_target_rotation = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(
                fac(target, random.uniform() * std::numbers::pi * 2, controls, control_values));
        }
    };
    auto single_target_rotation2 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls,
                                control_values));
        }
    };
    auto single_target_rotation3 = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target = shuffled[0];
            std::vector<std::uint64_t> controls(shuffled.begin() + 1, shuffled.begin() + 1 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                random.uniform() * std::numbers::pi * 2,
                                controls,
                                control_values));
        }
    };
    auto double_target = [&](auto fac) {
        for (auto nc : {0, 1, 2}) {
            std::vector<std::uint64_t> shuffled = random.permutation(n);
            std::uint64_t target0 = shuffled[0];
            std::uint64_t target1 = shuffled[1];
            std::vector<std::uint64_t> controls(shuffled.begin() + 2, shuffled.begin() + 2 + nc);
            std::vector<std::uint64_t> control_values = make_control_values(nc);
            gates.push_back(fac(target0, target1, controls, control_values));
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
                ComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        mat(i, j) = StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                    }
                }
                std::vector<std::uint64_t> control_values = make_control_values(nc);
                gates.push_back(fac(targets, mat, controls, control_values, false));
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
                SparseComplexMatrix mat(1 << nt, 1 << nt);
                for (auto i : std::views::iota(0, 1 << nt)) {
                    for (auto j : std::views::iota(0, 1 << nt)) {
                        if (random.uniform() < .5) {
                            mat.insert(i, j) =
                                StdComplex(random.uniform() * 2 - 1, random.uniform() * 2 - 1);
                        }
                    }
                }
                std::vector<std::uint64_t> control_values = make_control_values(nc);
                gates.push_back(fac(targets, mat, controls, control_values));
            }
        }
    };
    gates.push_back(gate::I<Prec, Space>());
    none_target_rotation(gate::GlobalPhase<Prec, Space>);
    single_target(gate::X<Prec, Space>);
    single_target(gate::Y<Prec, Space>);
    single_target(gate::Z<Prec, Space>);
    single_target(gate::H<Prec, Space>);
    single_target(gate::S<Prec, Space>);
    single_target(gate::Sdag<Prec, Space>);
    single_target(gate::T<Prec, Space>);
    single_target(gate::Tdag<Prec, Space>);
    single_target(gate::SqrtX<Prec, Space>);
    single_target(gate::SqrtXdag<Prec, Space>);
    single_target(gate::SqrtY<Prec, Space>);
    single_target(gate::SqrtYdag<Prec, Space>);
    single_target(gate::P0<Prec, Space>);
    single_target(gate::P1<Prec, Space>);
    single_target_rotation(gate::RX<Prec, Space>);
    single_target_rotation(gate::RY<Prec, Space>);
    single_target_rotation(gate::RZ<Prec, Space>);
    single_target_rotation(gate::U1<Prec, Space>);
    single_target_rotation2(gate::U2<Prec, Space>);
    single_target_rotation3(gate::U3<Prec, Space>);
    double_target(gate::Swap<Prec, Space>);
    dense_matrix(gate::DenseMatrix<Prec, Space>);
    sparse_matrix(gate::SparseMatrix<Prec, Space>);
    gates.push_back(
        gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("X 0 Y 2", random.uniform())));
    gates.push_back(gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("Z 0", random.uniform())));
    gates.push_back(
        gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("Z 3", random.uniform()), {1}));
    gates.push_back(
        gate::Pauli<Prec, Space>(PauliOperator<Prec, Space>("Z 1", random.uniform()), {0, 3}));
    gates.push_back(
        gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>("X 0 Y 2", random.uniform()),
                                         random.uniform() * std::numbers::pi * 2));
    gates.push_back(
        gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>("Z 0", random.uniform()),
                                         random.uniform() * std::numbers::pi * 2));
    gates.push_back(
        gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>("Z 3", random.uniform()),
                                         random.uniform() * std::numbers::pi * 2,
                                         {1}));
    gates.push_back(
        gate::PauliRotation<Prec, Space>(PauliOperator<Prec, Space>("Z 1", random.uniform()),
                                         random.uniform() * std::numbers::pi * 2,
                                         {0, 3}));
    for (auto&& g1 : gates) {
        for (auto&& g2 : gates) {
            auto state1 = StateVector<Prec, Space>::Haar_random_state(n);
            auto state2 = state1.copy();
            auto [mg, phase] = merge_gate(g1, g2);
            g1->update_quantum_state(state1);
            g2->update_quantum_state(state1);
            mg->update_quantum_state(state2);
            state2.multiply_coef(std::polar(1., phase));
            ASSERT_TRUE(same_state(state1, state2));
        }
    }
}

TYPED_TEST(MergeGateTest, MergeGate) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    merge_gate_test<Prec, Space>();
}
