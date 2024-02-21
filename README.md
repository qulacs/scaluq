# qulacs2023

Qulacs2023（正式名称未定）は、量子回路シミュレータ [Qulacs](https://github.com/qulacs/qulacs) をもとに再開発された、新しい Python/C++ ライブラリです。
大規模な量子回路、ノイズを伴う量子回路、パラメトリック量子回路の高速シミュレーションを実行することができます。
本ライブラリは、MITライセンスの下で公開しています。

[Qulacs](https://github.com/qulacs/qulacs) に比べ、以下の点が改善されています。

- [Kokkos](https://github.com/kokkos/kokkos) をベースとした実装により、実行環境(CPU/GPU) の切り替えを容易に行うことができます。切り替えの際にコードを変更する必要はありません。
- 同じ量子回路で、よりよい実行速度を実現します。
- ポインタをユーザから隠蔽したことにより、より安全に、簡単に記述できます。
- [nanobind](https://github.com/wjakob/nanobind) の導入により、よりコンパクトかつ高速な Python へのバインディングを実現します。

## 依存ライブラリ

- GCC 11 以上
- CMake 3.21 以上
- CUDA 12.2 以上（GPU利用時のみ）

## C++ ライブラリとしてのインストール

Qulacs2023 を静的ライブラリとしてインストールするには、以下の一連のコマンドを実行します。

```txt
git clone https://github.com/Qulacs-Osaka/qulacs2023
cd qulacs2023
./script/build_gcc.sh
```

NVIDIA GPU と CUDA が利用可能ならば、以下のコマンドで GPU バージョンをインストールできます。ビルドスクリプトの実行の際に `QULACS_USE_CUDA` オプションを付けます。

```txt
QULACS_USE_CUDA=ON ./script/build_gcc.sh
```

ただし、オプションを変更して再ビルドする際には、CMake にセットされたキャッシュ変数をクリアするため、必ず以下のコマンドを実行してください。

```txt
./script/clean.sh
```

## Python へのインストール

```txt
git clone https://github.com/Qulacs-Osaka/qulacs2023
cd qulacs2023
pip install .
```

### サンプルコード(Python)

```Python
from qulacs2023 import *
import math

def main():
    n_qubits = 3
    state = StateVector.Haar_random_state(n_qubits, 0)

    circuit = Circuit(n_qubits)
    circuit.add_gate(X(0))
    circuit.add_gate(CNOT(0, 1))
    circuit.add_gate(Y(1))
    circuit.add_gate(RX(1, math.pi / 2))
    circuit.update_quantum_state(state)

    observable = Operator(n_qubits)
    observable.add_random_operator(1, 0)
    value = observable.get_expectation_value(state)
    print(value)

initialize(InitializationSettings().set_num_threads(8))
main()
finalize()
```

### サンプルコード(C++)

```cpp
#include <iostream>

#include <circuit/circuit.hpp>
#include <gate/gate_factory.hpp>
#include <operator/operator.hpp>
#include <state/state_vector.hpp>

int main() {
    qulacs::initialize();  // must be called before using any qulacs methods
    {
        const int n_qubits = 3;
        qulacs::StateVector state = qulacs::StateVector::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        qulacs::Circuit circuit(n_qubits);
        circuit.add_gate(qulacs::X(0));
        circuit.add_gate(qulacs::CNOT(0, 1));
        circuit.add_gate(qulacs::Y(1));
        circuit.add_gate(qulacs::RX(1, M_PI / 2));
        circuit.update_quantum_state(state);

        qulacs::Operator observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    qulacs::finalize();  // must be called last
}
```
