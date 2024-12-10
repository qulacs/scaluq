#pragma once

#include <random>
#include <ranges>

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

    [[nodiscard]] std::vector<std::uint64_t> permutation(std::uint64_t n) {
        std::vector<std::uint64_t> shuffled(n);
        std::iota(shuffled.begin(), shuffled.end(), 0ULL);
        for (std::uint64_t i : std::views::iota(0ULL, n) | std::views::reverse) {
            std::uint64_t j = int32() % (i + 1);
            if (i != j) std::swap(shuffled[i], shuffled[j]);
        }
        return shuffled;
    }
};
}  // namespace scaluq
