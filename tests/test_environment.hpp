#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

using namespace std::complex_literals;

namespace scaluq {
class TestEnvironment : public testing::Environment {
    void SetUp() override { Kokkos::initialize(); }
    void TearDown() override { Kokkos::finalize(); }
};

inline auto kokkos_environment_this_variable_is_not_used =
    testing::AddGlobalTestEnvironment(new TestEnvironment());
}  // namespace scaluq
