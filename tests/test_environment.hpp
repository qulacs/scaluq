#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

namespace scaluq {
class TestEnvironment : public ::testing::Environment {
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

inline auto kokkos_environment_this_variable_is_not_used =
    testing::AddGlobalTestEnvironment(new TestEnvironment());

struct F16Test {
    constexpr static Precision Prec = Precision::F16;
    static std::string name() { return "F16"; }
};
struct F32Test {
    constexpr static Precision Prec = Precision::F32;
    static std::string name() { return "F32"; }
};
struct F64Test {
    constexpr static Precision Prec = Precision::F64;
    static std::string name() { return "F64"; }
};
struct BF16Test {
    constexpr static Precision Prec = Precision::BF16;
    static std::string name() { return "BF16"; }
};

template <typename List>
struct AddF16 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT16
template <typename... Types>
struct AddF16<::testing::Types<Types...>> {
    using Type = ::testing::Types<F16Test, Types...>;
};
#endif
template <typename List>
struct AddF32 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT32
template <typename... Types>
struct AddF32<::testing::Types<Types...>> {
    using Type = ::testing::Types<F32Test, Types...>;
};
#endif
template <typename List>
struct AddF64 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT64
template <typename... Types>
struct AddF64<::testing::Types<Types...>> {
    using Type = ::testing::Types<F64Test, Types...>;
};
#endif
template <typename List>
struct AddBF16 {
    using Type = List;
};
#ifdef SCALUQ_BFLOAT16
template <typename... Types>
struct AddBF16<::testing::Types<Types...>> {
    using Type = ::testing::Types<BF16Test, Types...>;
};
#endif
using TestTypes = AddF16<AddF32<AddF64<AddBF16<::testing::Types<>>::Type>::Type>::Type>::Type;
class NameGenerator {
public:
    template <typename T>
    static std::string GetName(int) {
        return T::name();
    }
};
template <typename T>
class FixtureBase : public ::testing::Test {
public:
    constexpr static Precision Prec = T::Prec;
};
template <typename T, typename Param>
class ParamFixtureBase : public ::testing::TestWithParam<Param> {
public:
    constexpr static Precision Prec = T::Prec;
};
}  // namespace scaluq
