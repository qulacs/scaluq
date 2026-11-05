#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
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
    std::vector result(batch_size, std::vector<std::uint64_t>(count));
    if (!count || !batch_size) return result;
    Kokkos::View<double**, Sp> cumulative("sampling_cumulative", batch_size, dim + 1);
    Kokkos::View<double*, Sp> totals("sampling_totals", batch_size);
    Kokkos::parallel_for(
        "sampling_cumulative",
        Kokkos::TeamPolicy<Sp>(batch_size, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<Sp>::member_type& team) {
            const auto batch = team.league_rank();
            Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, dim),
                                  [&](std::uint64_t i, double& update, bool final) {
                                      const double p = probability(batch, i);
                                      update += p >= 0. && p <= std::numeric_limits<double>::max()
                                                    ? p
                                                    : std::numeric_limits<double>::quiet_NaN();
                                      if (final) cumulative(batch, i + 1) = update;
                                  });
            team.team_barrier();
            Kokkos::single(Kokkos::PerTeam(team), [&] { totals(batch) = cumulative(batch, dim); });
        });
    auto upper = convert_view_to_vector<double, Space>(totals);
    for (double& total : upper) {
        if (!std::isfinite(total) || total <= 0.) {
            throw std::invalid_argument(
                "Sampling requires finite nonnegative probabilities with a positive total.");
        }
        total = std::nextafter(total, 0.);
    }
    const auto limits = convert_vector_to_view<double, Space>(upper);
    Kokkos::View<std::uint64_t*, Sp> samples("samples", batch_size * count);
    Kokkos::Random_XorShift64_Pool<Sp> pool(resolve_seed(seed));
    Kokkos::parallel_for(
        "sampling_choose",
        Kokkos::RangePolicy<Sp>(0, batch_size * count),
        KOKKOS_LAMBDA(std::uint64_t idx) {
            const auto batch = idx / count;
            auto generator = pool.get_state();
            const double r = generator.drand(0., limits(batch));
            std::uint64_t lo = 0, hi = dim + 1;
            while (hi - lo > 1) {
                const auto mid = (lo + hi) / 2;
                if (cumulative(batch, mid) > r)
                    hi = mid;
                else
                    lo = mid;
            }
            samples(idx) = lo;
            pool.free_state(generator);
        });
    const auto host_samples = convert_view_to_vector<std::uint64_t, Space>(samples);
    for (std::uint64_t b = 0; b < batch_size; ++b) {
        std::copy_n(host_samples.begin() + b * count, count, result[b].begin());
    }
    return result;
}
}  // namespace scaluq::internal
