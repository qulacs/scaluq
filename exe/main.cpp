#include <gtest/gtest.h>

#include <Eigen/Core>
#include <functional>
#include <iostream>
#include <types.hpp>

#include "./batch.hpp"

using namespace scaluq;
using namespace std;

template <typename Layout>
double run(UINT n_q, UINT batch, UINT loop = 6) {
    std::cout << "n_qubits: " << n_q << ", batch: " << batch << std::endl;
    Kokkos::Timer tm;
    tm.reset();
    for (UINT i = 0; i < loop; ++i) {
        StateVectorBatched<Layout> states(batch, n_q);
        states.set_state_vector(StateVector::Haar_random_state(n_q, std::rand()));
        auto ent = states.get_entropy();
        auto nrm = states.get_squared_norm();
        auto prb = states.get_zero_probability(n_q / 2);
    }
    return tm.seconds() / loop;
}

double run_serial(UINT n_q, UINT batch, UINT loop = 6) {
    std::cout << "n_qubits: " << n_q << ", batch: " << batch << std::endl;
    Kokkos::Timer tm;
    tm.reset();
    for (UINT i = 0; i < loop; ++i) {
        for (UINT b = 0; b < batch; ++b) {
            auto state = StateVector::Haar_random_state(n_q, 0);
            auto ent = state.get_entropy();
            auto nrm = state.get_squared_norm();
            auto prb = state.get_zero_probability(n_q / 2);
        }
    }
    return tm.seconds() / loop;
}

double run_serial_all_memory(UINT n_q, UINT batch, UINT loop = 6) {
    std::cout << "n_qubits: " << n_q << ", batch: " << batch << std::endl;
    Kokkos::Timer tm;
    tm.reset();
    for (UINT i = 0; i < loop; ++i) {
        std::vector<StateVector> states(batch);
        for (UINT b = 0; b < batch; ++b) states[b] = StateVector::Haar_random_state(n_q, 0);
        for (auto &state : states) auto ent = state.get_entropy();
        for (auto &state : states) auto nrm = state.get_squared_norm();
        for (auto &state : states) auto prb = state.get_zero_probability(n_q / 2);
    }
    return tm.seconds() / loop;
}

int main() {
    Kokkos::initialize();

    const UINT n_lim = 18, batch_lim = 3;
    std::vector left(n_lim + 1, std::vector<double>(batch_lim + 1, -1));
    std::vector right(n_lim + 1, std::vector<double>(batch_lim + 1, -1));
    std::vector serial(n_lim + 1, std::vector<double>(batch_lim + 1, -1));
    std::vector serial_all_mem(n_lim + 1, std::vector<double>(batch_lim + 1, -1));
    for (UINT n = 1; n <= n_lim; ++n) {
        for (UINT b = 0; b <= batch_lim; ++b) {
            const UINT batch_size = [](UINT pw) {
                UINT ret = 1;
                for (UINT i = 0; i < pw; ++i) ret *= 10;
                return ret;
            }(b);
            left[n][b] = run<Kokkos::LayoutLeft>(n, batch_size);
            right[n][b] = run<Kokkos::LayoutRight>(n, batch_size);
            serial[n][b] = run_serial(n, batch_size);
            serial_all_mem[n][b] = run_serial_all_memory(n, batch_size);
        }
    }

    std::cout << "LayoutLeft:\n";
    for (UINT n = 1; n <= n_lim; ++n) {
        for (auto d : left[n]) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "LayoutRight:\n";
    for (UINT n = 1; n <= n_lim; ++n) {
        for (auto d : right[n]) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Serial:\n";
    for (UINT n = 1; n <= n_lim; ++n) {
        for (auto d : serial[n]) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Serial(all states are on memory):\n";
    for (UINT n = 1; n <= n_lim; ++n) {
        for (auto d : serial_all_mem[n]) {
            std::cout << d << " ";
        }
        std::cout << std::endl;
    }

    Kokkos::finalize();
}
