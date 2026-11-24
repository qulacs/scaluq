#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <scaluq/util/random.hpp>
#include <scaluq/util/utility.hpp>

namespace scaluq::internal {
// Accumulate and sample in double so low-precision rounding cannot produce an
// out-of-range sample. A positive finite total also permits unnormalized states.
template <ExecutionSpace Space, typename Probability>
std::vector<std::vector<std::uint64_t>> sample_probabilities(std::uint64_t batch_size,
                                                             std::uint64_t dim,
                                                             std::uint64_t count,
                                                             std::optional<std::uint64_t> seed,
                                                             Probability probability) {
    using Sp = SpaceType<Space>;
    using Policy = Kokkos::TeamPolicy<Sp>;
    std::vector result(batch_size, std::vector<std::uint64_t>(count));
    if (!count || !batch_size) return result;

    // Aim for enough teams to parallelize large states even when the batch is small.
    constexpr std::uint64_t target_teams = 128;
    const auto blocks_per_state = 1 + (target_teams - 1) / batch_size;
    const auto desired_width = 1 + (dim - 1) / blocks_per_state;
    const auto block_width =
        std::min(dim, std::bit_ceil(std::clamp<std::uint64_t>(desired_width, 1024, 16384)));
    const auto block_count = 1 + (dim - 1) / block_width;

    Kokkos::View<double**, Kokkos::LayoutRight, Sp> cumulative(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sampling_cumulative"),
        batch_size,
        dim + 1);
    Kokkos::View<double**, Kokkos::LayoutRight, Sp> block_offsets(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sampling_block_offsets"),
        batch_size,
        block_count);
    Kokkos::parallel_for(
        "sampling_cumulative",
        Policy(batch_size * block_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Policy::member_type& team) {
            const auto batch = team.league_rank() / block_count;
            const auto block = team.league_rank() % block_count;
            const auto begin = block * block_width;
            const auto end = Kokkos::min(begin + block_width, dim);
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, begin, end),
                                  [&](std::uint64_t i, double& update, bool final) {
                                      const double p = probability(batch, i);
                                      update += p >= 0. && p <= std::numeric_limits<double>::max()
                                                    ? p
                                                    : std::numeric_limits<double>::quiet_NaN();
                                      if (final) cumulative(batch, i + 1) = update;
                                  });
        });
    Kokkos::View<double*, Sp> totals(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sampling_totals"), batch_size);
    Kokkos::View<double*, Sp> limits(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sampling_limits"), batch_size);
    Kokkos::parallel_for(
        "sampling_block_offsets",
        Policy(batch_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Policy::member_type& team) {
            const auto batch = team.league_rank();
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, block_count),
                                  [&](std::uint64_t block, double& update, bool final) {
                                      const auto offset = update;
                                      const auto end = Kokkos::min((block + 1) * block_width, dim);
                                      update += cumulative(batch, end);
                                      if (final) {
                                          block_offsets(batch, block) = offset;
                                          if (block + 1 == block_count) {
                                              totals(batch) = update;
                                              limits(batch) = Kokkos::nextafter(update, 0.);
                                          }
                                      }
                                  });
        });
    const auto host_totals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, totals);
    for (std::uint64_t batch = 0; batch < batch_size; ++batch) {
        const double total = host_totals(batch);
        if (!std::isfinite(total) || total <= 0.) {
            throw std::invalid_argument(
                "Sampling requires finite nonnegative probabilities with a positive total.");
        }
    }
    Kokkos::View<std::uint64_t*, Sp> samples(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "samples"), batch_size * count);
    Kokkos::Random_XorShift64_Pool<Sp> pool(resolve_seed(seed));
    Kokkos::parallel_for(
        "sampling_choose",
        Kokkos::RangePolicy<Sp>(0, batch_size * count),
        KOKKOS_LAMBDA(std::uint64_t idx) {
            const auto batch = idx / count;
            auto generator = pool.get_state();
            const double r = generator.drand(0., limits(batch));

            std::uint64_t first_block = 0, last_block = block_count;
            while (last_block - first_block > 1) {
                const auto mid = (first_block + last_block) / 2;
                if (block_offsets(batch, mid) > r)
                    last_block = mid;
                else
                    first_block = mid;
            }

            const double offset = block_offsets(batch, first_block);
            auto lo = first_block * block_width;
            auto hi = Kokkos::min(lo + block_width, dim);
            while (hi - lo > 1) {
                const auto mid = (lo + hi) / 2;
                if (offset + cumulative(batch, mid) > r)
                    hi = mid;
                else
                    lo = mid;
            }
            samples(idx) = lo;
            pool.free_state(generator);
        });
    const auto host_samples = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, samples);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        std::copy_n(host_samples.data() + b * count, count, result[b].begin());
    }
    return result;
}
}  // namespace scaluq::internal
