# Scaluq

![](https://app.readthedocs.org/projects/scaluq/badge/)
[![Build and Test](https://github.com/qulacs/scaluq/actions/workflows/test_all.yml/badge.svg)](https://github.com/qulacs/scaluq/actions/workflows/test_all.yml)
[![Format](https://github.com/qulacs/scaluq/actions/workflows/format.yml/badge.svg)](https://github.com/qulacs/scaluq/actions/workflows/format.yml)
[![Install to System](https://github.com/qulacs/scaluq/actions/workflows/install_merge.yml/badge.svg)](https://github.com/qulacs/scaluq/actions/workflows/install_merge.yml)
[![Sdist build](https://github.com/qulacs/scaluq/actions/workflows/sdist.yml/badge.svg)](https://github.com/qulacs/scaluq/actions/workflows/sdist.yml)
[![Wheel build](https://github.com/qulacs/scaluq/actions/workflows/wheel_merge.yml/badge.svg)](https://github.com/qulacs/scaluq/actions/workflows/wheel_merge.yml)

**英語版の README は [README.md](README.md) を参照してください。**

Scaluq は、量子回路シミュレータ [Qulacs](https://github.com/qulacs/qulacs) をもとに再開発された、新しい Python/C++ ライブラリです。
大規模な量子回路、ノイズを伴う量子回路、パラメトリック量子回路の高速シミュレーションを実行することができます。
本ライブラリは、MITライセンスの下で公開されています。

[Qulacs](https://github.com/qulacs/qulacs) に比べ、以下の点が改善されています。

- [Kokkos](https://github.com/kokkos/kokkos) をベースとした実装により、実行環境(CPU/GPU) の切り替えを容易に行うことができます。切り替えの際にコードを変更する必要はありません。
- CPU実行においては同等の実行速度を実現し、GPU実行においては同等以上の実行速度を実現します。
- ポインタをユーザから隠蔽したことにより、より安全に、簡単に記述できます。
- [nanobind](https://github.com/wjakob/nanobind) の導入によって、より軽量かつ低オーバーヘッドな Python バインディングを実現します。
- 複数の量子状態に対して、同じ構造を持ちパラメータのみが異なる量子回路を一括実行するためのバッチ実行機能を提供します。

# ドキュメント

https://scaluq.readthedocs.io/en/latest/index.html をご確認ください。

# パフォーマンス

量子回路シミュレーションの実行時間を、複数の既存量子回路シミュレータと比較しました。  
本ベンチマークでは、CX、RX、RZゲートをターゲット量子ビットを変えながら順に適用する回路を実行し、その平均実行時間を測定しました。

[ベンチマークのリポジトリ](https://github.com/Qulacs-Osaka/benchmark-scaluq) をご確認ください。

## 単一状態ベクトル更新 (2026年1月)

| CPU 結果 | GPU 結果 |
| ------- | --------|
| ![Single State Vector Update (CPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/multiple-gate/multithread/image/circuit.png) | ![Single State Vector Update (GPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/multiple-gate/gpu/image/circuit.png) |

## バッチ状態ベクトル更新 (2026年5月)

| バッチサイズを変化させた場合 (#qubits=16) | 量子ビット数を変化させた場合 (batch size=100) |
| -------------------------------------- | ------------------------------------------- |
| ![Batched State Vector Update (batch sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/batch/image/batch_sweep.png) | ![Batched State Vector Update (qubits sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/batch/image/qubits_sweep.png) |

## ビルド時要件

- Ninja 1.10 以上
- GCC 13 以上 または LLVM Clang 13 以上
  - CUDA 利用時はGCC 11以上が利用できるが、Clangは利用不可
- CMake 3.24 以上
- CUDA 12.8 以上（CUDA利用時のみ）
- IntelLLVM (SYCL利用時のみ)
  - Intel oneAPI DPC++/C++ Compiler (CC=icx/CXX=icpx)
- Python 3.10 以上 (Python利用時のみ)

CUDA を利用する場合は、使用する CUDA がサポートするホストコンパイラのバージョンを使用してください（CUDA Installation Guide の Host Compiler Support Policy を参照）。

※これより低いバージョンでも動作する可能性はありますが確認していません

※SYCLはCPU上での動作は確認できていますが、Intel GPUやNvidia GPUでの動作確認を行っていません。

## 実行時要件

- CUDA 12.8 以上（CUDA利用時のみ）
- SYCL
    - intel-level-zero-gpu
    - intel-opencl-icd
    - level-zero

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
|`SCALUQ_USE_SYCL`|`OFF`|GPU (SYCL)での並列処理を行うか|
|`SCALUQ_CPU_NATIVE`|`ON`| ビルダーのCPUアーキテクチャでビルドするか|
|`SCALUQ_CPU_ARCH`|-| ターゲットとなるCPUアーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_CPU_ARCH=SKX`)|
|`SCALUQ_CUDA_ARCH`|(自動識別)|`SCALUQ_USE_CUDA=ON`の場合、ターゲットとなるNvidia GPU アーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_CUDA_ARCH=AMPERE80`)|
|`SCALUQ_SYCL_ARCH`|(自動識別)|`SCALUQ_USE_SYCL=ON`の場合、ターゲットとなるIntel GPU アーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_SYCL_ARCH=INTEL_PVC`)|
|`SCALUQ_USE_TEST`|`OFF`|`test/`をビルドターゲットに含める。`ctest --test-dir build/`でテストのビルド・実行ができます|
|`SCALUQ_USE_EXE`|`OFF`|`exe/`をビルドターゲットに含める。`ninja -C build`でビルドしたあと、`build/exe/main` を実行してインストールなしで試せます|
|`SCALUQ_FLOAT16`|`OFF`|`f16`精度を有効にする|
|`SCALUQ_FLOAT32`|`ON`|`f32`精度を有効にする|
|`SCALUQ_FLOAT64`|`ON`|`f64`精度を有効にする|
|`SCALUQ_BFLOAT16`|`OFF`|`bf16`精度を有効にする|

## C++ ライブラリとしてインストール

Scaluq を静的ライブラリとしてインストールするには、以下の一連のコマンドを実行します。

```txt
git clone https://github.com/qulacs/scaluq
cd scaluq
script/configure
sudo -E env "PATH=$PATH" ninja -C build install
```

- 依存ライブラリのEigenとKokkosも同時にインストールされます
- `CMAKE_INSTALL_PREFIX`を設定することで `/usr/local/`以外にインストールすることもできます。ユーザーローカルにインストールしたい場合や、別の設定でビルドしたKokkosと衝突させたくない場合は明示的に指定してください。例: `CMAKE_INSTALL_PREFIX=~/.local script/configure; ninja -C build install`
- ビルドしたものを`/usr/local/`に配置するため`sudo`コマンドを用いていますが、ビルド時の環境変数をユーザーのものにするため例では`-E`と`env "PATH=$PATH"`を指定しています。
- NVIDIA GPU と CUDA が利用可能ならば、`SCALUQ_USE_CUDA=ON`を設定してconfigureすることでCUDAを利用するライブラリとしてインストールできます。例: `SCALUQ_USE_CUDA=ON script/configure; sudo env -E "PATH=$PATH" ninja -C build install`
- Intel GPU と SYCL が利用可能ならば、`SCALUQ_USE_SYCL=ON`を設定してconfigureすることでSYCLを利用するライブラリとしてインストールできます。例: `SCALUQ_USE_SYCL=ON script/configure; sudo env -E "PATH=$PATH" ninja -C build install`

オプションを変更して再ビルドする際には、CMake にセットされたキャッシュ変数をクリアするため、必ず以下のコマンドを実行してください。

```txt
rm build/CMakeCache.txt
```

インストール済みのScaluqを利用したプロジェクトでのCMake設定例を[example_project/](example_project/CMakeLists.txt)に提示しています。

## Python ライブラリとしてインストール

Python のライブラリとしても使用することができます。

```txt
pip install scaluq
```

GPUを利用する場合や`f32` `f64`以外の精度のサポートが必要な場合は、リポジトリをクローンしたのちにオプションを指定してインストールします。
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
#include <cstdint>
#include <iostream>

#include <scaluq/circuit/circuit.hpp>
#include <scaluq/gate/gate_factory.hpp>
#include <scaluq/operator/operator.hpp>
#include <scaluq/state/state_vector.hpp>

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        constexpr scaluq::Precision Prec = scaluq::Precision::F64;
        constexpr scaluq::ExecutionSpace Space = scaluq::ExecutionSpace::Default;
        const std::uint64_t n_qubits = 3;
        scaluq::StateVector<Prec, Space> state =
            scaluq::StateVector<Prec, Space>::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        scaluq::Circuit<Prec> circuit;
        circuit.add_gate(scaluq::gate::X<Prec>(0));
        circuit.add_gate(scaluq::gate::CNot<Prec>(0, 1));
        circuit.add_gate(scaluq::gate::Y<Prec>(1));
        circuit.add_gate(scaluq::gate::RX<Prec>(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        std::vector<scaluq::PauliOperator<Prec>> terms;
        terms.emplace_back(1, 0);
        scaluq::Operator<Prec, Space> observable(terms);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

`scaluq/all.hpp` を include すると、`SCALUQ_OMIT_TEMPLATE` を使ってテンプレート引数を省略できます。

```cpp
#include <cstdint>
#include <iostream>

#include <scaluq/all.hpp>

namespace my_scaluq {
SCALUQ_OMIT_TEMPLATE(scaluq::Precision::F64, scaluq::ExecutionSpace::Default)
}

using namespace my_scaluq;

int main() {
    scaluq::initialize();  // must be called before using any scaluq methods
    {
        const std::uint64_t n_qubits = 3;
        StateVector state = StateVector::Haar_random_state(n_qubits, 0);
        std::cout << state << std::endl;

        Circuit circuit;
        circuit.add_gate(gate::X(0));
        circuit.add_gate(gate::CNot(0, 1));
        circuit.add_gate(gate::Y(1));
        circuit.add_gate(gate::RX(1, std::numbers::pi / 2));
        circuit.update_quantum_state(state);

        std::vector<PauliOperator> terms;
        terms.emplace_back(1, 0);
        Operator observable(terms);
        auto value = observable.get_expectation_value(state);
        std::cout << value << std::endl;
    }
    scaluq::finalize();  // must be called last
}
```

## サンプルコード(Python)

```python
from scaluq import StateVector, Circuit, PauliOperator, Operator
from scaluq import gate
import math

n_qubits = 3
state = StateVector.Haar_random_state(n_qubits, 0)

circuit = Circuit()
circuit.add_gate(gate.X(0))
circuit.add_gate(gate.CNot(0, 1))
circuit.add_gate(gate.Y(1))
circuit.add_gate(gate.RX(1, math.pi / 2))
circuit.update_quantum_state(state)

terms = [PauliOperator("Z 0")]
observable = Operator(terms)
value = observable.get_expectation_value(state)
print(value)
```

# 精度と実行スペースの指定について

Scaluqでは、計算に使用する浮動小数点数のサイズとして`f16` `f32` `f64` `bf16`が選択できます。ただしデフォルトのビルドオプションでは`f16` `bf16`は無効になっています。
通常は`f64`の使用が推奨されますが、量子機械学習での利用などあまり精度が必要でない場合は`f32`以下を使用すると最大2~4倍の高速化が見込めます。

各精度の指定方法と内容は以下のとおりです。
|精度|C++で指定するテンプレート引数|Pythonのキーワード引数 (`precision=`)|内容|
|-|-|-|-|
|`f16`|`Precision::F16`|`'f16'`|IEEE754 binary16|
|`f32`|`Precision::F32`|`'f32'`|IEEE754 binary32|
|`f64`|`Precision::F64`|`'f64'`|IEEE754 binary64|
|`bf16`|`Precision::BF16`|`'bf16'`|bfloat16|

注意：`f16` / `bf16` 精度では、計算誤差が非常に大きくなる可能性があります（場合によっては $0.1$ を超えることもあります）。これらのオプションについては精度の検証を行っていません。

実行スペースは、計算を CPU と GPU のどちらで実行するかを決めます。

各実行スペースの指定方法と内容は以下のとおりです。

|実行スペース|C++で指定するテンプレート引数|Pythonのキーワード引数 (`space=`)|内容|
|-|-|-|-|
|`default`|`ExecutionSpace::Default`|`'default'`|CUDA、SYCL が有効なら GPU、そうでなければ CPU で実行|
|`host`|`ExecutionSpace::Host`|`'host'`|常に CPU で実行|
|`host_serial`|`ExecutionSpace::HostSerial`|`'host_serial'`|常に CPU で逐次実行|

同じ精度、実行スペースのオブジェクト同士でしか演算を行うことができません。
例えば32bit用に作成したゲートでは64bitの`StateVector`を更新できず、同じ精度でもhost用に作成したゲートではdefault用に作成した`StateVector`を更新できません。

C++の場合、状態、ゲート、演算子、回路のクラスやゲートを生成する関数が、テンプレート引数を取るようになっており、そこに`Precision`型の値と`ExecutionSpace`型の値を指定します。

Pythonの場合、`StateVector`や`Circuit`などのトップレベルのクラス、および`scaluq.gate`のゲートファクトリ関数は、`precision`と`space`のキーワード引数を受け取ります（デフォルトはそれぞれ`'f64'`と`'default'`）。

```py
from scaluq import StateVector, Circuit
from scaluq import gate

prec = 'f64'
space = 'default'

state = StateVector(3, precision=prec, space=space)
x = gate.X(0, precision=prec)
x.update_quantum_state(state)
print(state)
```
