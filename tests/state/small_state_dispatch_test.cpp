#include <gtest/gtest.h>

#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/kokkos.hpp>

#include "../test_environment.hpp"
#include "../util/util.hpp"

using namespace scaluq;
template <typename T>
class SmallStateDispatchTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(SmallStateDispatchTest, TestTypes, NameGenerator);

TYPED_TEST(SmallStateDispatchTest, ThresholdMatchesRegularKernels) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    struct Restore {
        std::uint64_t old = get_small_state_serial_threshold();
        ~Restore() { set_small_state_serial_threshold(old); }
    } restore;
    auto original = StateVector<Prec, Space>::Haar_random_state(5, 99);
    auto run = [&](std::uint64_t threshold) {
        set_small_state_serial_threshold(threshold);
        auto state = original.copy();
        gate::H<Prec>(0)->update_quantum_state(state);
        gate::X<Prec>(2, {4}, {0})->update_quantum_state(state);
        gate::RX<Prec>(1, 0.42, {4}, {1})->update_quantum_state(state);
        gate::Swap<Prec>(1, 3, {4}, {0})->update_quantum_state(state);
        ComplexMatrix matrix = ComplexMatrix::Identity(4, 4);
        matrix(0, 0) = -1.;
        gate::DenseMatrix<Prec, Space>({0, 2}, matrix, {4}, {1})->update_quantum_state(state);
        PauliOperator<Prec> p({0, 3}, {1, 2}, 1.0);
        gate::Pauli(p, {4}, {1})->update_quantum_state(state);
        gate::PauliRotation(p, 0.31, {4}, {0})->update_quantum_state(state);
        return state.get_amplitudes();
    };
    auto expected = run(0);
    for (auto threshold : {31ULL, 32ULL, 64ULL}) {
        auto result = run(threshold);
        ASSERT_EQ(result.size(), expected.size());
        for (std::size_t i = 0; i < result.size(); ++i) check_near<Prec>(result[i], expected[i]);
    }
    EXPECT_EQ(get_small_state_serial_threshold(), 64);
}

TYPED_TEST(SmallStateDispatchTest, BatchedThresholdMatchesRegularKernels) {
    constexpr Precision Prec = TestFixture::Prec;
    constexpr ExecutionSpace Space = TestFixture::Space;
    struct Restore {
        std::uint64_t old = get_small_state_serial_threshold();
        ~Restore() { set_small_state_serial_threshold(old); }
    } restore;
    auto original = StateVectorBatched<Prec, Space>::Haar_random_state(3, 4, false, 14);
    auto run = [&](std::uint64_t threshold) {
        set_small_state_serial_threshold(threshold);
        auto states = original.copy();
        gate::H<Prec>(0)->update_quantum_state(states);
        gate::X<Prec>(2, {3}, {0})->update_quantum_state(states);
        PauliOperator<Prec> p({0, 1}, {1, 2}, 1.0);
        gate::Pauli(p, {3}, {1})->update_quantum_state(states);
        gate::PauliRotation(p, 0.27, {3}, {0})->update_quantum_state(states);
        return states.get_amplitudes();
    };
    auto expected = run(0);
    for (auto threshold : {47ULL, 48ULL, 64ULL}) {
        auto result = run(threshold);
        for (std::size_t b = 0; b < result.size(); ++b)
            for (std::size_t i = 0; i < result[b].size(); ++i)
                check_near<Prec>(result[b][i], expected[b][i]);
    }
}
