#pragma once

#include <random>

#include "../types.hpp"

namespace scaluq {
class Random {
    std::mt19937_64 mt;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;

public:
    Random(std::uint64_t seed = std::random_device()())
        : mt(seed), uniform_dist(0, 1), normal_dist(0, 1) {}

    [[nodiscard]] double uniform() { return this->uniform_dist(this->mt); }

    [[nodiscard]] double normal() { return this->normal_dist(this->mt); }

    [[nodiscard]] std::uint64_t int64() { return this->mt(); }

    [[nodiscard]] std::uint32_t int32() { return this->mt() % UINT32_MAX; }
};
}  // namespace scaluq
