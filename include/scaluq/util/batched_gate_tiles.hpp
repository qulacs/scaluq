#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>

namespace scaluq::internal {

struct BatchedGateTiles {
    std::uint64_t width;
    std::uint64_t count;

    BatchedGateTiles(std::uint64_t batch_size, std::uint64_t basis_count) {
        constexpr std::uint64_t target_teams = 128;
        const auto desired_count = 1 + (target_teams - 1) / std::max(batch_size, std::uint64_t{1});
        const auto desired_width = basis_count / desired_count + (basis_count % desired_count != 0);

        // A gate operates over a power-of-two basis range, so every tile has the same width.
        width = std::min(std::max(basis_count, std::uint64_t{1}),
                         std::bit_ceil(std::clamp<std::uint64_t>(desired_width, 1024, 16384)));
        count = basis_count / width;
    }
};

}  // namespace scaluq::internal
