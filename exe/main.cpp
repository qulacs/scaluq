#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace nlohmann;

int main() {
    scaluq::initialize();  // 初期化

    constexpr Precision Real = Precision::F64;
    constexpr ExecutionSpace Exec = ExecutionSpace::Default;

    const std::uint64_t nqubits = 10;
    const std::uint64_t shots = 1000;

    // タイマー開始
    auto start_time = std::chrono::high_resolution_clock::now();

    {
        StateVector<Real, Exec> state(nqubits);

        Circuit<Real, Exec> circuit(nqubits);

        for (std::int64_t i = 0; i < 50; ++i) {
            if (rand() % 2 == 0) {
                std::uint64_t target = rand() % nqubits;
                std::uint64_t control = rand() % nqubits;
                if (target == control) ++control;
                circuit.add_gate(gate::Z<Real, Exec>(target, std::vector<std::uint64_t>{control}));
            } else {
                std::uint64_t target = rand() % nqubits;
                std::uint64_t control = rand() % nqubits;
                if (target == control) ++control;
                auto z = gate::Z<Real, Exec>(target, std::vector<std::uint64_t>{control});
                auto x = gate::X<Real, Exec>(target, std::vector<std::uint64_t>{control});
                auto pgate = gate::Probabilistic<Real, Exec>({0.7, 0.3}, {x, z});
                circuit.add_gate(pgate);
            }
        }

        // ノイズありシミュレーションを実行
        auto result = circuit.simulate_noise(state, shots);

        // 結果の表示
        std::cout << "=== Benchmark: Probabilistic Gate Simulation ===\n";
        std::cout << "Shots: " << shots << ", Qubits: " << nqubits << "\n";
        std::cout << "Number of distinct outputs: " << result.size() << "\n\n";

        // for (std::size_t i = 0; i < result.size(); ++i) {
        //     const auto& [state_vec, count] = result[i];
        //     std::cout << "[Outcome " << i << "] Count: " << count << "\n";
        //     std::cout << state_vec << "\n";
        // }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    std::cout << "\nTotal simulation time: " << duration.count() << " seconds\n";

    scaluq::finalize();  // 終了処理
}
