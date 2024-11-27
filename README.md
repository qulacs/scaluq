# scaluq

scaluq は、量子回路シミュレータ [Qulacs](https://github.com/qulacs/qulacs) をもとに再開発された、新しい Python/C++ ライブラリです。
大規模な量子回路、ノイズを伴う量子回路(未実装)、パラメトリック量子回路の高速シミュレーションを実行することができます。
本ライブラリは、MITライセンスの下で公開されています。

[Qulacs](https://github.com/qulacs/qulacs) に比べ、以下の点が改善されています。

- [Kokkos](https://github.com/kokkos/kokkos) をベースとした実装により、実行環境(CPU/GPU) の切り替えを容易に行うことができます。切り替えの際にコードを変更する必要はありません。
- よりよい実行速度を実現します。
- ポインタをユーザから隠蔽したことにより、より安全に、簡単に記述できます。
- [nanobind](https://github.com/wjakob/nanobind) の導入により、よりコンパクトかつ高速な Python へのバインディングを実現します。
- 複数の量子状態に対して同じ回路を適用させるようなケースに対して、より高速なインターフェースを提供します（未実装）。

## ビルド時要件

- Ninja 1.10 以上
- GCC 11 以上
- CMake 3.21 以上
- CUDA 12.6 以上（GPU利用時のみ）
※これより低いバージョンでも動作する可能性はありますが確認していません

## 実行時要件
- CUDA 12.6 以上（GPU利用時のみ）
※これより低いバージョンでも動作する可能性はありますが確認していません

## C++ ライブラリとしてインストール

scaluq を静的ライブラリとしてインストールするには、以下の一連のコマンドを実行します。

```txt
git clone https://github.com/qulacs/scaluq
cd scaluq
script/configure
sudo -E env "PATH=$PATH" ninja -C build install
```

- 依存ライブラリのEigenとKokkosも同時にインストールされます
- `CMAKE_INSTALL_PREFIX`を設定することで `/usr/local`以外にインストールすることもできます。ユーザーローカルにインストールしたい場合や、別の設定でビルドしたKokkosと衝突させたくない場合は明示的に指定してください。例: `CMAKE_INSTALL_PREFIX=~/.local script/configure; ninja -C build install`
- ビルドしたものを`/usr/local/bin`に配置するため`sudo`コマンドを用いていますが、ビルド時の環境変数をユーザーのものにするため例では`-E`と`env "PATH=$PATH"`を指定しています。
- NVIDIA GPU と CUDA が利用可能ならば、`SCALUQ_USE_CUDA=Yes`を設定してconfigureすることでCUDAを利用するライブラリとしてインストールできます。例: `SCALUQ_USE_CUDA=Yes script/configure; sudo env -E "PATH=$PATH" ninja -C build install'`

オプションを変更して再ビルドする際には、CMake にセットされたキャッシュ変数をクリアするため、必ず以下のコマンドを実行してください。

```txt
rm build/CMakeCache.txt
```

インストール済みのscaluqを利用したプロジェクトでのCMake設定例を[example_project/](example_project/CMakeLists.txt)に提示しています。

## Python ライブラリとしてインストール
Python のライブラリとしても使用することができます。
```txt
pip install scaluq
```

GPUを利用する場合は、リポジトリをクローンしたのちにインストールします。
```txt
git clone https://github.com/qulacs/scaluq
cd ./scaluq
SCALUQ_USE_CUDA=ON pip install . 
```

## Python ドキュメント
Python ライブラリとしてインストールした際の、関数の説明や型の情報がまとめられている、簡易的なドキュメントを用意しています。以下のリンクから確認できます。
https://scaluq.readthedocs.io/en/latest/index.html


## サンプルコード(C++)

```cpp
#include <iostream>
#include <cstdint>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/operator/operator.hpp>
#include <scaluq/state/state_vector.hpp>

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector state = scaluq::StateVector::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X(0));
        circuit.add_gate(scaluq::gate::CNot(0, 1));
        circuit.add_gate(scaluq::gate::Y(1));
        circuit.add_gate(scaluq::gate::RX(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        scaluq::Operator observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

## サンプルコード(Python)

```Python
from scaluq import *
import math

n_qubits = 3
state = StateVector.Haar_random_state(n_qubits, 0)

circuit = Circuit(n_qubits)
circuit.add_gate(gate::X(0))
circuit.add_gate(gate::CNot(0, 1))
circuit.add_gate(gate::Y(1))
circuit.add_gate(gate::RX(1, math.pi / 2))
circuit.update_quantum_state(state)

observable = Operator(n_qubits)
observable.add_random_operator(1, 0)
value = observable.get_expectation_value(state)
print(value)

```
