#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>
#include <scaluq/util/random.hpp>
#include <scaluq/util/utility.hpp>

namespace scaluq::internal {
// SplitMix64 finalizer used to derive a uniform draw for each shot.
KOKKOS_INLINE_FUNCTION std::uint64_t mix_sampling_bits(std::uint64_t bits) {
    bits = (bits ^ (bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
    bits = (bits ^ (bits >> 27)) * 0x94d049bb133111ebULL;
    return bits ^ (bits >> 31);
}

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
    if (!count || !batch_size) return std::vector<std::vector<std::uint64_t>>(batch_size);
    if (dim == 0) {
        throw std::invalid_argument(
            "Sampling requires finite nonnegative probabilities with a positive total.");
    }
    // The result buffer also stores one validation flag per batch.
    if (count >= std::vector<std::uint64_t>().max_size() / batch_size) {
        throw std::length_error("Too many samples requested.");
    }

    // Aim for enough teams to parallelize large states even when the batch is small.
    constexpr std::uint64_t target_teams = 128;
    const auto blocks_per_state = 1 + (target_teams - 1) / batch_size;
    const auto desired_width = 1 + (dim - 1) / blocks_per_state;
    const auto amplitudes_per_block =
        std::min(dim, std::bit_ceil(std::clamp<std::uint64_t>(desired_width, 1024, 16384)));
    // Reduce the prefix table when scanning many more amplitudes than we draw.
    const auto amplitudes_per_bin =
        std::clamp<std::uint64_t>(std::bit_floor(dim / count), 1, 16);
    const auto bin_count = 1 + (dim - 1) / amplitudes_per_bin;
    const auto bins_per_block = 1 + (amplitudes_per_block - 1) / amplitudes_per_bin;
    const auto block_count = 1 + (bin_count - 1) / bins_per_block;

    const auto offsets_begin = bin_count + 1;
    const auto limit_index = offsets_begin + block_count;
    Kokkos::View<double**, Kokkos::LayoutRight, Sp> tables(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "sampling_tables"),
        batch_size,
        limit_index + 1);
    Kokkos::parallel_for(
        "sampling_cumulative",
        Policy(batch_size * block_count, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Policy::member_type& team) {
            const auto batch = team.league_rank() / block_count;
            const auto block = team.league_rank() % block_count;
            const auto begin = block * bins_per_block;
            const auto end = Kokkos::min(begin + bins_per_block, bin_count);
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, begin, end),
                                  [&](std::uint64_t bin, double& update, bool final) {
                                      const auto begin = bin * amplitudes_per_bin;
                                      const auto end = Kokkos::min(begin + amplitudes_per_bin, dim);
                                      for (auto i = begin; i < end; ++i) {
                                          const double p = probability(batch, i);
                                          update += p < 0. ? std::numeric_limits<double>::quiet_NaN()
                                                           : p;
                                      }
                                      if (final) tables(batch, bin + 1) = update;
                                  });
        });
    const auto total_shots = batch_size * count;
    Kokkos::View<std::uint64_t*, Sp> samples(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "samples"), total_shots + batch_size);
    Kokkos::parallel_for(
        "sampling_block_offsets",
        Policy(batch_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Policy::member_type& team) {
            const auto batch = team.league_rank();
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, block_count),
                                  [&](std::uint64_t block, double& update, bool final) {
                                      const auto offset = update;
                                      const auto end = Kokkos::min((block + 1) * bins_per_block, bin_count);
                                      update += tables(batch, end);
                                      if (final) {
                                          tables(batch, offsets_begin + block) = offset;
                                          if (block + 1 == block_count) {
                                              const bool valid =
                                                  update > 0. &&
                                                  update <= std::numeric_limits<double>::max();
                                              samples(total_shots + batch) = valid;
                                              tables(batch, limit_index) = valid ? Kokkos::nextafter(update, 0.) : 0.;
                                          }
                                      }
                                  });
        });
    constexpr std::uint64_t rng_stride = 0x9e3779b97f4a7c15ULL;
    const auto key = mix_sampling_bits(resolve_seed(seed) + rng_stride);
    Kokkos::parallel_for(
        "sampling_choose",
        Kokkos::RangePolicy<Sp>(0, total_shots),
        KOKKOS_LAMBDA(std::uint64_t idx) {
            const auto batch = idx / count;
            const auto random = mix_sampling_bits(key + rng_stride * (idx + 1));
            const double r = static_cast<double>(random >> 11) * 0x1p-53 * tables(batch, limit_index);

            std::uint64_t first_block = 0, last_block = block_count;
            while (last_block - first_block > 1) {
                const auto mid = (first_block + last_block) / 2;
                if (tables(batch, offsets_begin + mid) > r)
                    last_block = mid;
                else
                    first_block = mid;
            }

            const double offset = tables(batch, offsets_begin + first_block);
            const auto block_begin = first_block * bins_per_block;
            auto lo = block_begin;
            auto hi = Kokkos::min(lo + bins_per_block, bin_count);
            while (hi - lo > 1) {
                const auto mid = (lo + hi) / 2;
                if (offset + tables(batch, mid) > r)
                    hi = mid;
                else
                    lo = mid;
            }
            const auto first_basis = lo * amplitudes_per_bin;
            const auto last_basis = Kokkos::min(first_basis + amplitudes_per_bin, dim) - 1;
            double local_cdf = lo == block_begin ? 0. : tables(batch, lo);
            auto basis = first_basis;
            for (; basis < last_basis; ++basis) {
                local_cdf += probability(batch, basis);
                if (offset + local_cdf > r) break;
            }
            // Roundoff between bin prefixes and the local scan can leave a zero-weight tail.
            if (basis == last_basis) {
                while (basis > first_basis && probability(batch, basis) == 0.) --basis;
            }
            samples(idx) = basis;
        });
    const auto host_samples = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, samples);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        if (!host_samples(total_shots + b)) {
            throw std::invalid_argument(
                "Sampling requires finite nonnegative probabilities with a positive total.");
        }
    }
    std::vector<std::vector<std::uint64_t>> result;
    result.reserve(batch_size);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        const auto begin = host_samples.data() + b * count;
        result.emplace_back(begin, begin + count);
    }
    return result;
}
}  // namespace scaluq::internal
