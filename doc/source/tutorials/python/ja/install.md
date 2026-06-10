# ScaluqをPythonパッケージとしてインストールする方法

## PyPIからインストール
Scaluqを簡単にインストールするには以下を実行してください。

```python
pip install scaluq
```

配布されたパッケージの設定については以下のとおりです。

- 並列実行にはOpenMPを使用してください。
- すべてのシミュレーションはCPU上で行われます。
    - GPUは使えません。
    - 実行スペースの`default`と`host`は同じです。
- 精度は`f32`と`f64`が利用できます。

## ソースコードからインストール
特定のコミットハッシュやデフォルトでないオプションでインストールしたい場合は、ソースコードからScaluqをインストールすることができます。

ビルドの要件は以下のとおりです。

- Ninja ≥ 1.10
- GCC ≥ 13 or LLVM Clang ≥ 18
  - CUDAを有効化した場合、GCC ≥ 11は利用できますが、Clangは使用できません。
- CMake ≥ 3.24
- CUDA ≥ 12.8 (CUDA利用時のみ)
- Python ≥ 3.10 (Python利用時のみ)

CUDAの利用時には、お使いのCUDA toolkitでサポートされているホストコンパイラのバージョンを使ってください。
詳しくはCUDA Installation Guide Host Compiler Support Policyをご確認ください。

注意：提示したビルド要件よりも低いバージョンでも動作する可能性はありますが、動作確認は行われていないためビルドの要件に記載のバージョンでビルドすることを推奨します。

ビルドオプションは以下のとおりです。

|変数名|デフォルト値|意味|
|-|-|-|
|`CMAKE_C_COMPILER`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)|
|`CMAKE_CXX_COMPILER`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)|
|`CMAKE_BUILD_TYPE`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)|
|`CMAKE_INSTALL_PREFIX`|-|See [CMake Document](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)|
|`SCALUQ_USE_OMP`|`ON`|CPUでの並列処理にOpenMPを利用するか|
|`SCALUQ_USE_CUDA`|`OFF`|GPU (CUDA)での並列処理を行うか|
|`SCALUQ_CPU_NATIVE`|`ON`| ビルダーのCPUアーキテクチャでビルドするか|
|`SCALUQ_CPU_ARCH`|-| ターゲットとなるCPUアーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_CPU_ARCH=SKX`)|
|`SCALUQ_CUDA_ARCH`|(自動識別)|`SCALUQ_USE_CUDA=ON`の場合、ターゲットとなるNvidia GPU アーキテクチャ (名前は[Kokkos CMake Keywords](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html)を参照、例: `SCALUQ_CUDA_ARCH=AMPERE80`)|
|`SCALUQ_USE_TEST`|`OFF`|`test/`をビルドターゲットに含める。`ctest --test-dir build/`でテストのビルド・実行ができます|
|`SCALUQ_USE_EXE`|`OFF`|`exe/`をビルドターゲットに含める。`ninja -C build`でビルドしたあと、`build/exe/main` を実行してインストールなしで試せます|
|`SCALUQ_FLOAT16`|`OFF`|`f16`精度を有効にする|
|`SCALUQ_FLOAT32`|`ON`|`f32`精度を有効にする|
|`SCALUQ_FLOAT64`|`ON`|`f64`精度を有効にする|
|`SCALUQ_BFLOAT16`|`OFF`|`bf16`精度を有効にする|

Scaluqをインストールするには、まずgithubのリポジトリをクローンし、ディレクトリに入ってください。

```
git clone https://github.com/qulacs/scaluq
cd scaluq
```

そして、環境変数によって設定を行い、Scaluqをインストールしてください。

```
SCALUQ_USE_CUDA=ON SCALUQ_FLOAT32=OFF pip install .
```
