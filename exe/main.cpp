#include <iostream>
#include <scaluq/all.hpp>
using namespace scaluq;

using namespace scaluq;

int main() {
    initialize();  // must be called before using any scaluq methods
    {
        // correct
        double sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<2>>({0, 0}, {10, 10}),
            KOKKOS_LAMBDA(const int i, const int j, double& sum) {
                sum += static_cast<int>(i * 10 + j);
            },
            sum);
        std::cout << "Sum of products: " << sum << std::endl;  // 4950

        // correct
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(
                Kokkos::DefaultExecutionSpace(), 5, Kokkos::AUTO),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& team) {
                int team_sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadMDRange(team, 10, 10),
                    [&](const int i, const int j, int& sum) { sum += i * 10 + j; },
                    team_sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    // each value of team_sum is 4950
                    Kokkos::printf("Team %d sum: %d\n", team.league_rank(), team_sum);
                });
            });
        Kokkos::fence();

        // wrong
        Kokkos::parallel_for(
            Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>(
                Kokkos::DefaultHostExecutionSpace(), 5, Kokkos::AUTO),
            KOKKOS_LAMBDA(
                const Kokkos::TeamPolicy<Kokkos::DefaultHostExecutionSpace>::member_type& team) {
                int team_sum = 0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadMDRange(team, 10, 10),
                    [&](const int i, const int j, int& sum) { sum += i * 10 + j; },
                    team_sum);
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    // each value of team_sum is 450 ?
                    Kokkos::printf("Team %d sum: %d\n", team.league_rank(), team_sum);
                });
            });
        Kokkos::fence();
    }
    finalize();
}
