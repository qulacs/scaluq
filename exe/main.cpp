#include <chrono>
#include <iostream>
#include <scaluq/all.hpp>

using namespace scaluq;
using namespace std::chrono;

int main() {
    scaluq::initialize();

    constexpr std::size_t n_qubits = 10;
    constexpr std::size_t n_terms = 5;
    constexpr std::size_t n_iterations = 100;

    constexpr Precision Prec = Precision::F64;
    constexpr ExecutionSpace Space = ExecutionSpace::Default;

    // 初期状態
    StateVector<Prec, Space> state(n_qubits);
    state.set_zero_norm_state();  // 正規化状態に初期化

    // 適当な Operator を作る（パウリZの和）
    Operator<Prec, Space> op(n_qubits);
    for (std::size_t i = 0; i < n_terms; ++i) {
        op.add_operator(PauliOperator<Prec, Space>(n_qubits, i));  // i番目にZ
        op.add_term(pw, 1.0);                                      // 係数1.0で追加
    }

    // ベンチマーク
    double total_time_ms = 0.0;
    for (std::size_t i = 0; i < n_iterations; ++i) {
        StateVector<Prec, Space> s = state.copy();

        auto start = high_resolution_clock::now();
        op.apply_to_state(s);  // 計測対象
        auto end = high_resolution_clock::now();

        double elapsed_ms = duration<double, std::milli>(end - start).count();
        total_time_ms += elapsed_ms;
    }

    std::cout << "Average time over " << n_iterations << " runs: " << (total_time_ms / n_iterations)
              << " ms\n";

    scaluq::finalize();
    return 0;
}
