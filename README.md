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
- GCC 11 以上 (CUDAを利用しない場合13以上)
- CMake 3.21 以上
- CUDA 12.6 以上（CUDA利用時のみ）
- Python 3.9 以上 (Python利用時のみ)
※これより低いバージョンでも動作する可能性はありますが確認していません

## 実行時要件
- CUDA 12.6 以上（CUDA利用時のみ）
※これより低いバージョンでも動作する可能性はありますが確認していません

## ビルドオプション
ビルド時のオプションを`script/configure`や`pip install .`実行時の環境変数で指定できます。

|変数名|デフォルト値|意味|
|-|-|-|
|`CMAKE_C_COMPILER`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)|
|`CMAKE_CXX_COMPILER`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)|
|`CMAKE_BUILD_TYPE`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)|
|`CMAKE_INSTALL_PREFIX`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)|
|`SCALUQ_USE_OMP`|`ON`|CPUでの並列処理にOpenMPを利用するか|
|`SCALUQ_USE_CUDA`|`OFF`|GPU (CUDA)での並列処理を行うか|
|`SCALUQ_CUDA_ARCH`|(自動識別)|`CMAKE_USE_CUDA=ON`の場合、ターゲットとなるNvidia GPU アーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_CUDA_ARCH=AMPERE80`)|
|`SCALUQ_USE_TEST`|ON|`test/`をビルドターゲットに含める。`ninja -C build test`でテストのビルド/実行ができる|
|`SCALUQ_USE_EXE`|ON|`exe/`をビルドターゲットに含める。インストールせずに実行を試すことができ、`ninja -C build`でのビルド後、`exe/main.cpp`の内容を`build/exe/main`で実行できる。|
|`SCALUQ_FLOAT16`|OFF|`f16`精度を有効にする|
|`SCALUQ_FLOAT32`|ON|`f32`精度を有効にする|
|`SCALUQ_FLOAT64`|ON|`f64`精度を有効にする|
|`SCALUQ_BFLOAT16`|OFF|`bf16`精度を有効にする|

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
- NVIDIA GPU と CUDA が利用可能ならば、`SCALUQ_USE_CUDA=ON`を設定してconfigureすることでCUDAを利用するライブラリとしてインストールできます。例: `SCALUQ_USE_CUDA=ON script/configure; sudo env -E "PATH=$PATH" ninja -C build install'`

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

GPUを利用する場合や`f32` `f64`精度のサポートが必要な場合は、リポジトリをクローンしたのちにオプションを指定してインストールします。
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
        constexpr Precision Prec = Precision::F64;
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector<Prec> state = scaluq::StateVector<Prec>::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit<Prec> circuit(n_qubits);
        circuit.add_gate(scaluq::gate::X<Prec>(0));
        circuit.add_gate(scaluq::gate::CNot<Prec>(0, 1));
        circuit.add_gate(scaluq::gate::Y<Prec>(1));
        circuit.add_gate(scaluq::gate::RX<Prec>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        scaluq::Operator<Prec> observable(n_qubits);
        observable.add_random_operator(1, 0);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

## サンプルコード(Python)

```Python
from scaluq.f64 import *
import math

n_qubits = 3
state = StateVector.Haar_random_state(n_qubits, 0)

circuit = Circuit(n_qubits)
circuit.add_gate(gate.X(0))
circuit.add_gate(gate.CNot(0, 1))
circuit.add_gate(gate.Y(1))
circuit.add_gate(gate.RX(1, math.pi / 2))
circuit.update_quantum_state(state)

observable = Operator(n_qubits)
observable.add_random_operator(1, 0)
value = observable.get_expectation_value(state)
print(value)

```

# 精度指定について
scaluqでは、計算に使用する浮動小数点数のサイズとして`f16` `f32` `f64` `bf16`が選択できます。ただしデフォルトのビルドオプションでは`f16` `bf16`は無効になっています。
通常は`f64`の使用が推奨されますが、量子機械学習での利用などあまり精度が必要でない場合は`f32`以下を使用すると最大2~4倍の高速化が見込めます。

各精度の指定方法と内容は以下のとおりです。
|精度|C++で指定するテンプレート引数|Pythonで指定するサブモジュールの名前|内容|
|-|-|-|-|
|`f16`|`Precision::F16`|`f16`|IEEE754 binary16|
|`f32`|`Precision::F32`|`f16`|IEEE754 binary32|
|`f64`|`Precision::F64`|`f16`|IEEE754 binary64|
|`bf16`|`Precision::BF16`|`bf16`|bfloat16|

同じ精度のオブジェクト同士でしか演算を行うことができません。
例えば32bit用に作成したゲートでは64bitの`StateVector`を更新できません。

C++の場合、状態、ゲート、演算子、回路のクラスやゲートを生成する関数が、テンプレート引数を取るようになっており、そこに`Precision`型の値を指定します。

Pythonの場合、精度に合わせて`scaluq.f32`と`scaluq.f64`のどちらかのサブモジュールからオブジェクトを`import`します。
Pythonでは以下のように`importlib`を用いることで文字列からダイナミックに選択できます。
```py
import importlib

prec = 'f64'
scaluq_sub = importlib.import_module(f'scaluq.{prec}')
StateVector = scaluq_sub.StateVector
gate = scaluq_sub.gate

state = StateVector(3)
x = gate.X(0)
x.update_quantum_state(state)
print(state)
```
