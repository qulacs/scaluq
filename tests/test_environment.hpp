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

template <Precision _Prec, ExecutionSpace _Space>
struct TestType {
    constexpr static Precision Prec = _Prec;
    constexpr static ExecutionSpace Space = _Space;
    static std::string name() {
        std::string ret;
        if constexpr (Prec == Precision::F16)
            ret += "F16";
        else if constexpr (Prec == Precision::F32)
            ret += "F32";
        else if constexpr (Prec == Precision::F64)
            ret += "F64";
        else if constexpr (Prec == Precision::BF16)
            ret += "BF16";

        if constexpr (Space == ExecutionSpace::Host)
            ret += "HostSpace";
        else if constexpr (Space == ExecutionSpace::Default)
            ret += "DefaultSpace";
        return ret;
    }
};

template <Precision Prec, typename... Types>
struct AddPrecision {
    using Type =
        std::conditional_t<std::is_same_v<internal::SpaceType<ExecutionSpace::Host>,
                                          internal::SpaceType<ExecutionSpace::Default>>,
                           ::testing::Types<TestType<Prec, ExecutionSpace::Host>, Types...>,
                           ::testing::Types<TestType<Prec, ExecutionSpace::Host>,
                                            TestType<Prec, ExecutionSpace::Default>,
                                            Types...>>;
};

template <typename List>
struct AddF16 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT16
template <typename... Types>
struct AddF16<::testing::Types<Types...>> {
    using Type = typename AddPrecision<Precision::F16, Types...>::Type;
};
#endif
template <typename List>
struct AddF32 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT32
template <typename... Types>
struct AddF32<::testing::Types<Types...>> {
    using Type = typename AddPrecision<Precision::F32, Types...>::Type;
};
#endif
template <typename List>
struct AddF64 {
    using Type = List;
};
#ifdef SCALUQ_FLOAT64
template <typename... Types>
struct AddF64<::testing::Types<Types...>> {
    using Type = typename AddPrecision<Precision::F64, Types...>::Type;
};
#endif
template <typename List>
struct AddBF16 {
    using Type = List;
};
#ifdef SCALUQ_BFLOAT16
template <typename... Types>
struct AddBF16<::testing::Types<Types...>> {
    using Type = typename AddPrecision<Precision::BF16, Types...>::Type;
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
    constexpr static ExecutionSpace Space = T::Space;
};
}  // namespace scaluq
