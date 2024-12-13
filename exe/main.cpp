#include <chrono>
#include <iostream>
#include <random>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/state/state_vector.hpp>
#include <stdfloat>

int main() {
    Kokkos::initialize();
    std::mt19937 mt(0);
    constexpr std::uint64_t n_qubits = 20;
    std::uniform_int_distribution<std::uint64_t> dist(0, n_qubits - 1);
    {
        using Fp = scaluq::F16;
        scaluq::StateVector<Fp> state(n_qubits);
        auto st = std::chrono::system_clock::now();
        for (int i = 0; i < 10000; i++) {
            auto x_gate = scaluq::gate::X<Fp>(dist(mt));
            x_gate->update_quantum_state(state);
        }
        auto ed = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count()
                  << std::endl;
    }
    {
        using Fp = scaluq::F32;
        scaluq::StateVector<Fp> state(n_qubits);
        auto st = std::chrono::system_clock::now();
        for (int i = 0; i < 10000; i++) {
            auto x_gate = scaluq::gate::X<Fp>(dist(mt));
            x_gate->update_quantum_state(state);
        }
        auto ed = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count()
                  << std::endl;
    }
    {
        using Fp = scaluq::F64;
        scaluq::StateVector<Fp> state(n_qubits);
        auto st = std::chrono::system_clock::now();
        for (int i = 0; i < 10000; i++) {
            auto x_gate = scaluq::gate::X<Fp>(dist(mt));
            x_gate->update_quantum_state(state);
        }
        auto ed = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count()
                  << std::endl;
    }
    {
        using Fp = scaluq::BF16;
        scaluq::StateVector<Fp> state(n_qubits);
        auto st = std::chrono::system_clock::now();
        for (int i = 0; i < 10000; i++) {
            auto x_gate = scaluq::gate::X<Fp>(dist(mt));
            x_gate->update_quantum_state(state);
        }
        auto ed = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(ed - st).count()
                  << std::endl;
    }
    Kokkos::finalize();
}
