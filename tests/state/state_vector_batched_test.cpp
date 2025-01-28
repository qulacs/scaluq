#include <gtest/gtest.h>

#include <scaluq/state/state_vector_batched.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;

#define FLOAT_AND_SPACE(Fp, Sp) template <std::floating_point Fp, ExecutionSpace Sp>
#define EXECUTE_TEST(Name, arg1, arg2)                \
    TEST(StateVectorBatchedTest, Name) {              \
        Test##Name<double, DefaultSpace>(arg1, arg2); \
        Test##Name<double, CPUSpace>(arg1, arg2);     \
        Test##Name<float, DefaultSpace>(arg1, arg2);  \
        Test##Name<float, CPUSpace>(arg1, arg2);      \
    }

FLOAT_AND_SPACE(Fp, Sp)
void TestHaarRandomStateNorm(std::uint32_t batch_size, std::uint32_t n_qubits) {
    const auto states = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);
    auto norms = states.get_squared_norm();
    for (auto x : norms) ASSERT_NEAR(x, 1., eps<Fp>);
}
EXECUTE_TEST(HaarRandomStateNorm, 4, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestLoadAndAmplitues(std::uint32_t batch_size, std::uint32_t n_qubits) {
    const std::uint64_t dim = 1 << n_qubits;
    std::vector states_h(batch_size, std::vector<Complex<Fp>>(dim));
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < dim; ++i) {
            states_h[b][i] = b * dim + i;
        }
    }
    StateVectorBatched<Fp, Sp> states(batch_size, n_qubits);
    states.load(states_h);
    auto amps = states.get_amplitudes();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < dim; ++i) {
            ASSERT_EQ(amps[b][i].real(), b * dim + i);
        }
    }
}
EXECUTE_TEST(LoadAndAmplitues, 5, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestOperateState(std::uint32_t batch_size, std::uint32_t n_qubits) {
    auto states = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);
    auto states_add = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);
    const Complex<Fp> coef(2.1, 3.5);

    auto states_cp = states.copy();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        ASSERT_TRUE(same_state(states.get_state_vector_at(b), states_cp.get_state_vector_at(b)));
    }

    states_cp = states.copy();
    states.add_state_vector_with_coef(coef, states_add);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at(b);
        v.add_state_vector_with_coef(coef, states_add.get_state_vector_at(b));
        ASSERT_TRUE(same_state(v, states.get_state_vector_at(b)));
    }

    states_cp = states.copy();
    states.multiply_coef(coef);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto v = states_cp.get_state_vector_at(b);
        v.multiply_coef(coef);
        ASSERT_TRUE(same_state(v, states.get_state_vector_at(b)));
    }
}
EXECUTE_TEST(OperateState, 5, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestZeroProbs(std::uint32_t batch_size, std::uint32_t n_qubits) {
    auto states = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);
    for (std::uint64_t i = 0; i < n_qubits; ++i) {
        auto zero_probs = states.get_zero_probability(i);
        for (std::uint64_t b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector_at(b);
            ASSERT_NEAR(zero_probs[b], state.get_zero_probability(i), eps<Fp>);
        }
    }
}
EXECUTE_TEST(ZeroProbs, 5, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestMarginalProbs(std::uint32_t batch_size, std::uint32_t n_qubits) {
    auto states = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);

    Random rd(0);
    for (std::uint64_t i = 0; i < 10; ++i) {
        std::vector<std::uint64_t> targets;
        for (std::uint64_t j = 0; j < n_qubits; ++j) {
            targets.push_back(rd.int32() % 3);
        }
        auto mg_probs = states.get_marginal_probability(targets);
        for (std::uint64_t b = 0; b < batch_size; ++b) {
            auto state = states.get_state_vector_at(b);
            ASSERT_NEAR(mg_probs[b], state.get_marginal_probability(targets), eps<Fp>);
        }
    }
}
EXECUTE_TEST(MarginalProbs, 5, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestEntropy(std::uint32_t batch_size, std::uint32_t n_qubits) {
    auto states = StateVectorBatched<Fp, Sp>::Haar_random_state(batch_size, n_qubits, false);
    auto entropies = states.get_entropy();
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        auto state = states.get_state_vector_at(b);
        ASSERT_NEAR(entropies[b], state.get_entropy(), eps<Fp>);
    }
}
EXECUTE_TEST(Entropy, 5, 3)

FLOAT_AND_SPACE(Fp, Sp)
void TestSampling(std::uint32_t batch_size, std::uint32_t n_qubits) {
    StateVectorBatched<Fp, Sp> states(batch_size, n_qubits);
    std::vector vv(batch_size, std::vector<Complex<Fp>>(states.dim()));
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i < states.dim(); ++i) {
            vv[b][i] = i;
        }
    }
    states.load(vv);
    states.normalize();

    std::uint32_t n_sampling = 2048 * n_qubits;
    auto result = states.sampling(n_sampling);
    std::vector sampling_counts(batch_size, std::vector<std::uint32_t>(states.dim(), 0));

    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i + 1 < n_sampling; ++i) {
            ++sampling_counts[b][result[b][i]];
        }
    }
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        for (std::uint64_t i = 0; i + 1 < states.dim(); ++i) {
            ASSERT_LE(sampling_counts[b][i], sampling_counts[b][i + 1]);
        }
    }
}
EXECUTE_TEST(Sampling, 5, 3)
