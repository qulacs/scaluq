#include <gtest/gtest.h>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/gate/param_gate_factory.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

template <typename T>
class CircuitOptimizeTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(CircuitOptimizeTest, TestTypes, NameGenerator);

TYPED_TEST(CircuitOptimizeTest, Basic) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    constexpr std::uint64_t N = 10;
    constexpr std::uint64_t M = 100;
    Random random;
    Circuit<Prec, Space> circuit(N);
    std::vector<std::function<void()>> adders;
    std::vector<std::string> keys;
    adders.push_back([&] { circuit.add_gate(gate::I<Prec, Space>()); });
    adders.push_back([&] {
        circuit.add_gate(
            gate::GlobalPhase<Prec, Space>(random.uniform() * std::numbers::pi_v<double> * 2));
    });
    adders.push_back([&] {
        circuit.add_gate(
            gate::GlobalPhase<Prec, Space>(random.uniform() * std::numbers::pi_v<double> * 2,
                                           {random.int64() % N},
                                           {random.int32() & 1}));
    });
    auto gen_dense = [&](std::uint64_t num_targets, std::uint64_t num_controls) {
        auto perm = random.permutation(N);
        ComplexMatrix mat = ComplexMatrix::Identity(1, 1);
        for ([[maybe_unused]] std::uint64_t _ : std::views::iota(std::uint64_t{0}, num_targets)) {
            mat = internal::kronecker_product(mat, get_eigen_matrix_random_one_target_unitary());
        }
        std::vector<std::uint64_t> control_values(num_controls);
        for (std::uint64_t& val : control_values) val = random.int32() & 1;
        return gate::DenseMatrix<Prec, Space>(
            std::vector<std::uint64_t>(perm.begin(), perm.begin() + num_targets),
            mat,
            std::vector<std::uint64_t>(perm.begin() + num_targets,
                                       perm.begin() + num_targets + num_controls),
            control_values);
    };
    adders.push_back([&] { circuit.add_gate(gen_dense(1, 0)); });
    adders.push_back([&] { circuit.add_gate(gen_dense(1, 1)); });
    adders.push_back([&] { circuit.add_gate(gen_dense(1, 2)); });
    adders.push_back([&] { circuit.add_gate(gen_dense(2, 0)); });
    adders.push_back([&] { circuit.add_gate(gen_dense(2, 1)); });
    adders.push_back([&] { circuit.add_gate(gen_dense(2, 2)); });
    adders.push_back([&] {
        circuit.add_gate(gate::Probabilistic<Prec, Space>({1.}, {gate::I<Prec, Space>()}));
    });
    adders.push_back([&] {
        std::string new_key = std::to_string(keys.size());
        circuit.add_param_gate(gate::ParamRX<Prec, Space>(0), new_key);
        keys.push_back(new_key);
    });
    for ([[maybe_unused]] std::uint64_t _ : std::views::iota(std::uint64_t{0}, M)) {
        adders[random.int32() % adders.size()]();
    }
    std::map<std::string, double> params;
    for (const std::string& key : keys) {
        params[key] = random.uniform() * std::numbers::pi_v<double> * 2;
    }

    StateVector<Prec, Space> state0 = StateVector<Prec, Space>::Haar_random_state(N);
    StateVector<Prec, Space> state1 = state0.copy();
    circuit.update_quantum_state(state0, params);
    circuit.optimize();
    circuit.update_quantum_state(state1, params);
    assert(same_state(state0, state1));
}
