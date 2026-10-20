#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <scaluq/gate/gate_factory.hpp>

#include "../../src/gate/update_ops.hpp"
#include "../../src/gate/update_ops_matrix_8x8.hpp"
#include "../test_environment.hpp"

namespace {
using namespace scaluq;
using scaluq::internal::CoefKind;
using scaluq::internal::Complex;
using scaluq::internal::convert_internal_matrix_to_external_matrix;
using scaluq::internal::DenseMatrixGateImpl;
using scaluq::internal::GatePtr;
using scaluq::internal::Matrix8x8;
using scaluq::internal::three_target_dense_matrix_gate;
using scaluq::internal::three_target_dense_matrix_gate_scalar;

template <typename T>
class Matrix8x8Test : public FixtureBase<T> {};
TYPED_TEST_SUITE(Matrix8x8Test, TestTypes, NameGenerator);

template <int Pattern>
constexpr CoefKind matrix_kind(std::size_t index) {
    if constexpr (Pattern == 0) return CoefKind::General;
    if constexpr (Pattern == 1) return static_cast<CoefKind>(((index / 8) ^ (index % 8)) % 5);
    // Includes a completely zero row and mixed kinds within each SIMD coefficient.
    if constexpr (Pattern == 2)
        return index < 8 ? CoefKind::Zero : static_cast<CoefKind>((index / 8 * 3 + index % 8) % 5);
    return CoefKind::Zero;
}

template <int Pattern, Precision Prec>
auto make_matrix() {
    Matrix8x8<Prec> matrix;
    for (std::size_t r = 0; r < 8; ++r) {
        for (std::size_t c = 0; c < 8; ++c) {
            const auto kind = matrix_kind<Pattern>(r * 8 + c);
            const double real = (static_cast<int>((r * 7 + c * 3) % 17) - 8) / 16.;
            const double imag = (static_cast<int>((r * 5 + c * 11) % 19) - 9) / 16.;
            matrix[r][c] = kind == CoefKind::Zero   ? Complex<Prec>(0)
                           : kind == CoefKind::One  ? Complex<Prec>(1)
                           : kind == CoefKind::Real ? Complex<Prec>(real)
                           : kind == CoefKind::Imag ? Complex<Prec>(0, imag)
                                                    : Complex<Prec>(real, imag);
        }
    }
    return matrix;
}

template <int Pattern, typename State, std::size_t... I>
void apply_kernel(std::uint64_t targets,
                  std::uint64_t controls,
                  std::uint64_t values,
                  const Matrix8x8<State::prec>& matrix,
                  State& state,
                  bool scalar,
                  std::index_sequence<I...>) {
    if (scalar) {
        three_target_dense_matrix_gate_scalar<matrix_kind<Pattern>(I)...>(
            targets, controls, values, matrix, state);
    } else {
        three_target_dense_matrix_gate<matrix_kind<Pattern>(I)...>(
            targets, controls, values, matrix, state);
    }
}

template <Precision Prec>
std::vector<StdComplex> reference(const std::vector<StdComplex>& input,
                                  const std::array<unsigned, 3>& targets,
                                  std::uint64_t controls,
                                  std::uint64_t values,
                                  const Matrix8x8<Prec>& matrix) {
    auto output = input;
    const auto mask = (1ULL << targets[0]) | (1ULL << targets[1]) | (1ULL << targets[2]);
    for (std::size_t i = 0; i < input.size(); ++i) {
        if ((i & controls) != values) continue;
        const unsigned r = ((i >> targets[0]) & 1) | (((i >> targets[1]) & 1) << 1) |
                           (((i >> targets[2]) & 1) << 2);
        output[i] = 0;
        for (unsigned c = 0; c < 8; ++c) {
            const auto j = (i & ~mask) | ((c & 1ULL) << targets[0]) |
                           (((c >> 1) & 1ULL) << targets[1]) | (((c >> 2) & 1ULL) << targets[2]);
            output[i] += static_cast<StdComplex>(matrix[r][c]) * input[j];
        }
    }
    return output;
}

template <int Pattern, Precision Prec, ExecutionSpace Space, bool Batched>
void check_kernels() {
    using State =
        std::conditional_t<Batched, StateVectorBatched<Prec, Space>, StateVector<Prec, Space>>;
    const auto matrix = make_matrix<Pattern, Prec>();
    constexpr double tolerance = Prec == Precision::F64   ? 1e-12
                                 : Prec == Precision::F32 ? 2e-6
                                                          : 0.1;
    for (unsigned n = 3; n <= 6; ++n) {
        auto state = [n] {
            if constexpr (Batched)
                return State::uninitialized_state(3, n);
            else
                return State::uninitialized_state(n);
        }();
        std::vector<std::vector<StdComplex>> input(Batched ? 3 : 1,
                                                   std::vector<StdComplex>(1ULL << n));
        for (std::size_t b = 0; b < input.size(); ++b) {
            for (std::size_t i = 0; i < input[b].size(); ++i) {
                input[b][i] = StdComplex((static_cast<int>((i + b * 3) % 17) - 8) / 16.,
                                         (static_cast<int>((i * 3 + b) % 13) - 6) / 16.);
            }
        }
        for (unsigned q0 = 0; q0 < n; ++q0) {
            for (unsigned q1 = q0 + 1; q1 < n; ++q1) {
                for (unsigned q2 = q1 + 1; q2 < n; ++q2) {
                    const auto targets = (1ULL << q0) | (1ULL << q1) | (1ULL << q2);
                    const auto available = ((1ULL << n) - 1) ^ targets;
                    for (auto controls = available;; controls = (controls - 1) & available) {
                        for (auto values = controls;; values = (values - 1) & controls) {
                            for (bool scalar : {false, true}) {
                                if constexpr (Batched)
                                    state.load(input);
                                else
                                    state.load(input[0]);
                                apply_kernel<Pattern>(targets,
                                                      controls,
                                                      values,
                                                      matrix,
                                                      state,
                                                      scalar,
                                                      std::make_index_sequence<64>{});
                                const auto actual = [&] {
                                    if constexpr (Batched)
                                        return state.get_amplitudes();
                                    else
                                        return std::vector<std::vector<StdComplex>>{
                                            state.get_amplitudes()};
                                }();
                                for (std::size_t b = 0; b < input.size(); ++b) {
                                    const auto expected =
                                        reference(input[b], {q0, q1, q2}, controls, values, matrix);
                                    for (std::size_t i = 0; i < expected.size(); ++i) {
                                        ASSERT_LE(std::abs(actual[b][i] - expected[i]), tolerance)
                                            << "n=" << n << " targets=" << targets
                                            << " controls=" << controls << " values=" << values
                                            << " scalar=" << scalar << " batch=" << b
                                            << " index=" << i;
                                    }
                                }
                            }
                            if (values == 0) break;
                        }
                        if (controls == 0) break;
                    }
                }
            }
        }
    }
}

TYPED_TEST(Matrix8x8Test, GeneralAndSpecializedCoefficients) {
    constexpr auto P = TestFixture::Prec;
    constexpr auto S = TestFixture::Space;
    check_kernels<0, P, S, false>();
    check_kernels<1, P, S, false>();
    check_kernels<2, P, S, false>();
    check_kernels<3, P, S, false>();
    check_kernels<0, P, S, true>();
    check_kernels<1, P, S, true>();
}

TYPED_TEST(Matrix8x8Test, DenseMatrixIntegration) {
    constexpr auto P = TestFixture::Prec;
    constexpr auto S = TestFixture::Space;
    const auto fixed = make_matrix<0, P>();
    ComplexMatrix matrix(8, 8);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) matrix(r, c) = static_cast<StdComplex>(fixed[r][c]);
    // Non-sorted targets also exercise the factory's row/column reordering.
    auto gate = gate::DenseMatrix<P, S>({4, 0, 2}, matrix, {1, 5}, {0, 1});
    const auto restored = gate->get_matrix();
    auto state = StateVector<P, S>::Haar_random_state(6, 17);
    const auto input = state.get_amplitudes();
    Matrix8x8<P> ordered;
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) ordered[r][c] = restored(r, c);
    const auto expected =
        reference(input, {0, 2, 4}, (1ULL << 1) | (1ULL << 5), 1ULL << 5, ordered);
    gate->update_quantum_state(state);
    const auto actual = state.get_amplitudes();
    constexpr double tolerance = P == Precision::F64 ? 1e-12 : P == Precision::F32 ? 2e-6 : 0.1;
    for (std::size_t i = 0; i < expected.size(); ++i)
        ASSERT_LE(std::abs(actual[i] - expected[i]), tolerance);
    const GatePtr<DenseMatrixGateImpl<P, S>> impl = gate;
    const auto internal =
        convert_internal_matrix_to_external_matrix<P, S>(impl->get_matrix_internal());
    EXPECT_TRUE(internal.isApprox(restored));
}
}  // namespace
