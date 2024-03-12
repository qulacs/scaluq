#pragma once

#include <random>

#include "../types.hpp"

namespace qulacs {
class Random {
    std::mt19937_64 mt;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;

public:
    Random(UINT seed = std::random_device()());

    [[nodiscard]] double uniform();

    [[nodiscard]] double normal();

    [[nodiscard]] UINT int64();

    [[nodiscard]] std::uint32_t int32();
};
}  // namespace qulacs
