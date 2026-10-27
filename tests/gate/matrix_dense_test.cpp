#include <gtest/gtest.h>

#include <array>
#include <scaluq/gate/gate_factory.hpp>

#include "../../src/gate/update_ops.hpp"
#include "../test_environment.hpp"

namespace {
using namespace scaluq;
using scaluq::internal::convert_external_matrix_to_internal_matrix;
using scaluq::internal::multi_dense_matrix_gate;

template <typename T>
class GeneralDenseMatrixTest : public FixtureBase<T> {};
TYPED_TEST_SUITE(GeneralDenseMatrixTest, TestTypes, NameGenerator);

template <Precision P, ExecutionSpace S, bool Batched>
void check_general_dense() {
    using State = std::conditional_t<Batched, StateVectorBatched<P, S>, StateVector<P, S>>;
    constexpr unsigned qubits = 8;
    std::vector<std::vector<StdComplex>> input(Batched ? 3 : 1,
                                               std::vector<StdComplex>(1 << qubits));
    for (std::size_t b = 0; b < input.size(); ++b)
        for (std::size_t i = 0; i < input[b].size(); ++i)
            input[b][i] = StdComplex((static_cast<int>((i + b * 3) % 17) - 8) / 16.,
                                     (static_cast<int>((i * 3 + b) % 13) - 6) / 16.);
    auto state = [] {
        if constexpr (Batched)
            return State::uninitialized_state(3, qubits);
        else
            return State::uninitialized_state(qubits);
    }();
    // Small matrices check the generic entry point, which may also be used directly.
    // Four or more targets exercise the public gate's general-matrix dispatch.
    for (std::uint64_t targets : {0x0, 0x1, 0x5, 0x15, 0xf, 0xf0, 0x55, 0xdb, 0xff}) {
        std::vector<std::uint64_t> positions;
        for (unsigned q = 0; q < qubits; ++q)
            if (targets & (1ULL << q)) positions.push_back(q);
        const std::size_t dim = 1ULL << positions.size();
        ComplexMatrix matrix(dim, dim);
        for (std::size_t r = 0; r < dim; ++r)
            for (std::size_t c = 0; c < dim; ++c)
                matrix(r, c) = StdComplex((static_cast<int>((r * 5 + c * 3) % 17) - 8) / 64.,
                                          (static_cast<int>((r * 7 + c * 11) % 19) - 9) / 64.);
        const auto internal = convert_external_matrix_to_internal_matrix<P, S>(matrix);
        const std::uint64_t available = 0xff ^ targets;
        for (std::uint64_t controls :
             std::array<std::uint64_t, 3>{0, available & -available, available}) {
            for (std::uint64_t values :
                 std::array<std::uint64_t, 3>{0, controls, controls & 0xaaULL}) {
                std::vector<std::vector<StdComplex>> expected = input;
                for (std::size_t b = 0; b < input.size(); ++b) {
                    for (std::size_t i = 0; i < input[b].size(); ++i) {
                        if ((i & controls) != values) continue;
                        std::size_t row = 0;
                        for (std::size_t q = 0; q < positions.size(); ++q)
                            row |= ((i >> positions[q]) & 1ULL) << q;
                        expected[b][i] = 0;
                        for (std::size_t c = 0; c < dim; ++c) {
                            std::size_t index = i & ~targets;
                            for (std::size_t q = 0; q < positions.size(); ++q)
                                index |= ((c >> q) & 1ULL) << positions[q];
                            expected[b][i] += matrix(row, c) * input[b][index];
                        }
                    }
                }
                for (bool public_gate : {false, true}) {
                    if constexpr (Batched)
                        state.load(input);
                    else
                        state.load(input[0]);
                    if (public_gate) {
                        std::vector<std::uint64_t> control_positions, control_values;
                        for (unsigned q = 0; q < qubits; ++q) {
                            if (controls & (1ULL << q)) {
                                control_positions.push_back(q);
                                control_values.push_back((values >> q) & 1ULL);
                            }
                        }
                        gate::DenseMatrix<P, S>(
                            positions, matrix, control_positions, control_values)
                            ->update_quantum_state(state);
                    } else {
                        multi_dense_matrix_gate(targets, controls, values, internal, state);
                    }
                    const auto actual = [&] {
                        if constexpr (Batched)
                            return state.get_amplitudes();
                        else
                            return std::vector<std::vector<StdComplex>>{state.get_amplitudes()};
                    }();
                    constexpr double tolerance = P == Precision::F64   ? 1e-12
                                                 : P == Precision::F32 ? 2e-5
                                                                       : 0.1;
                    for (std::size_t b = 0; b < input.size(); ++b)
                        for (std::size_t i = 0; i < input[b].size(); ++i)
                            ASSERT_LE(std::abs(actual[b][i] - expected[b][i]), tolerance)
                                << "targets=" << targets << " controls=" << controls
                                << " values=" << values << " public=" << public_gate
                                << " batch=" << b << " index=" << i;
                }
            }
        }
    }
}

TYPED_TEST(GeneralDenseMatrixTest, TargetsControlsAndBatches) {
    check_general_dense<TestFixture::Prec, TestFixture::Space, false>();
    check_general_dense<TestFixture::Prec, TestFixture::Space, true>();
}
}  // namespace
