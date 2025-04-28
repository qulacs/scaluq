#include <ranges>
#include <scaluq/types.hpp>

namespace scaluq::internal {
ComplexMatrix get_expanded_matrix(const ComplexMatrix& from_matrix,
                                  const std::vector<std::uint64_t>& from_targets,
                                  std::uint64_t from_control_mask,
                                  std::uint64_t from_control_value_mask,
                                  std::vector<std::uint64_t>& to_operands) {
    std::vector<std::uint64_t> targets_map(from_targets.size());
    std::ranges::transform(from_targets, targets_map.begin(), [&](std::uint64_t x) {
        return std::ranges::lower_bound(to_operands, x) - to_operands.begin();
    });
    std::vector<std::uint64_t> idx_map(1ULL << from_targets.size());
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (std::uint64_t j : std::views::iota(0ULL, from_targets.size())) {
            idx_map[i] |= (i >> j & 1) << targets_map[j];
        }
    }
    std::uint64_t to_control_mask = 0;
    std::uint64_t to_control_value_mask = 0;
    for (std::uint64_t sub_mask = from_control_mask; sub_mask; sub_mask &= (sub_mask - 1)) {
        std::uint32_t ctrz = std::countr_zero(sub_mask);
        std::uint32_t sll = std::ranges::lower_bound(to_operands, ctrz) - to_operands.begin();
        to_control_mask |= 1ULL << sll;
        to_control_value_mask |= (from_control_value_mask >> ctrz & 1ULL) << sll;
    }

    std::vector<std::uint64_t> outer_indices;
    outer_indices.reserve(
        1ULL << (to_operands.size() - from_targets.size() - std::popcount(from_control_mask)));
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << to_operands.size())) {
        if ((i & to_control_mask) == to_control_value_mask) outer_indices.push_back(i);
    }
    ComplexMatrix to_matrix =
        ComplexMatrix::Zero(1ULL << to_operands.size(), 1ULL << to_operands.size());

    // 制御条件を満たすインデクス
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << from_targets.size())) {
        for (std::uint64_t j : std::views::iota(0ULL, 1ULL << from_targets.size())) {
            for (std::uint64_t o : outer_indices) {
                to_matrix(idx_map[i] | to_control_value_mask | o,
                          idx_map[j] | to_control_value_mask | o) = from_matrix(i, j);
            }
        }
    }
    // 制御条件を満たさないインデクス
    for (std::uint64_t i : std::views::iota(0ULL, 1ULL << to_operands.size())) {
        if ((i & to_control_mask) != to_control_value_mask) to_matrix(i, i) = 1;
    }
    return to_matrix;
}
}  // namespace scaluq::internal
