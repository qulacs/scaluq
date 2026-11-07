# F2 調査報告: Kokkos + AdaptiveCpp + Metal で scaluq を Apple Silicon GPU で動かせるか

調査日: 2026-11-07
対象: `ref/scaluq_improvements.md`（統合ブランチ側）の F2・F3 の手順 1〜2 と、手順 3 の机上評価
結論の扱い: 実機で確認した事実と、コードを読んで推定した事項は分けて書く。性能値は参考値（後述の理由で再測定が必要）。

## 結論

1. AdaptiveCpp（develop）の Metal バックエンドは、この Mac（M3 Ultra、macOS 26.7.1）で動いた。単純な SYCL カーネル、`sycl::reduction`、ローカルメモリ、group/sub_group 演算は正しい結果を返した。
2. Kokkos 5.2.0（scaluq が固定している版）の SYCL バックエンドは、**そのままでは acpp でビルドできない**。19 ファイル・約 120 行のローカルパッチでビルドは通った。
3. パッチ後の Kokkos を Metal GPU で動かした結果は次のとおり。
   - `parallel_for`、`Kokkos::Random`、`Kokkos::complex<float>` の `atomic_add` は正しく動いた。
   - **`parallel_reduce`（Range・MDRange・TeamPolicy）と `parallel_scan` は、ワークグループが複数になると誤った値を返した**。
   - 原因は AdaptiveCpp の Metal バックエンドにある。ワークグループ間でメモリの可視性が保証されない（素の SYCL で再現済み）。
   - scaluq は内積・ノルム・期待値・確率をすべて reduce で計算するので、現状では scaluq を正しく動かせない。
4. したがって、F2（第一候補）は**現時点では実用にならない**。ただし、障害は 2 つの具体的な不具合に絞れた。どちらも上流（AdaptiveCpp、Kokkos）に報告・修正を提案できる粒度になっている。
   - AdaptiveCpp Metal のメモリ順序の不具合
   - Kokkos の DPC++ 依存
5. 推奨: 上流 2 件の報告と並行して、F4 の案 B（基本カーネルを Metal で直接実装）を小さく試作して比較するのがよい。scaluq 側の f32 専用化の作業（double 除去など）は、どちらの案でも必要になる。

## 環境

| 項目 | 値 |
|---|---|
| マシン | Apple M3 Ultra（GPU 80 コア、`applegpu_g15d`）、メモリ 512 GB |
| OS | macOS 26.7.1（25G241）。Xcode 本体は無し、Command Line Tools のみ |
| LLVM | Homebrew `llvm@21` 21.1.8 と `lld@21`（keg-only）。AdaptiveCpp の generic SSCP は LLVM 15〜21 に対応 |
| AdaptiveCpp | develop `6c923f68`（2026-11-05）。`-DWITH_METAL_BACKEND=ON`、metal-cpp（Apple 配布の `metal-cpp_26.zip`） |
| Kokkos | 5.2.0（`d9397f95`、scaluq の `FetchContent` と同じ） |
| ビルド物の場所 | `~/tools/metal-f2/`（リポジトリ外） |

## AdaptiveCpp のビルドで必要だったこと

手順は公式文書 `doc/install-metal.md` のとおり。ただし macOS 固有の落とし穴が 4 つあった。

- `ACPP_COMPILER_FEATURE_PROFILE` の既定値は、macOS では `none` になる。この既定のままだと generic SSCP（Metal に必要）が無効になるため、`full` を明示する。公式は 2 段ビルド（LLVM に AdaptiveCpp を組み込む）を推奨しているが、brew の LLVM 21 に対する 1 段ビルドでも動いた。
- `ld64.lld` が必要。`lld@21` を入れて `-DACPP_LLD_PATH` で指定する。
- brew の clang は SDK を自動では見つけない。`SDKROOT=$(xcrun --show-sdk-path)` が必要。
  - 指定しないと `stdlib.h` の `ldiv_t` が未定義というエラーになる。
  - 既定の SDK（27.0）と OS（26.x）が食い違うと、リンク時に警告が出る。今回は `MacOSX26.sdk` を明示した。
- OpenCL バックエンドは自動で有効になり、macOS 付属の OpenCL 1.2 とリンクできずに失敗する。`-DWITH_OPENCL_BACKEND=OFF` で無効にする。

`acpp-info` で `Loaded backend 1: Metal / Found device: Apple M3 Ultra` が表示された。報告された主な値は次のとおり。

- `local_mem_size` 32 KiB、`sub_group_sizes` 32、`max_group_size` 1024
- USM は device と shared が可、`usm_atomic_shared_allocations` は 0
- double の扱いは報告されない（文書上、double は非対応）

## 実機検証の結果

### 素の SYCL（AdaptiveCpp → Metal）

テストは `ref/f2_metal/tests/` に置いた。

| 項目 | 結果 |
|---|---|
| `parallel_for`（device / shared USM、2^24 要素） | 正しい |
| 64 ビット整数の除算・剰余・`popcount`（scaluq のインデックス計算に相当） | 正しい |
| `sycl::reduction`（float） | 正しい（相対誤差 3.6e-8） |
| `local_accessor` + `group_barrier` による木構造リダクション | 正しい |
| `reduce_over_group`（group / sub_group）、`shift_group_left/right`、`permute_group_by_xor`、`select_from_group`、`group_broadcast` | 正しい |
| 32 ビット整数と float の `fetch_add` | 正しい |
| 64 ビット `fetch_add`、CAS ループ（1 アドレスに集中） | 結果は正しいが、競合時に極端に遅い。64 ビット `fetch_add` は 2^16 スレッドで約 2.1 秒、2^20 スレッドでは 30 秒以内に終わらない。64 ビット CAS ループは 1024 スレッドで約 0.66 秒、2^16 スレッドでは 20 秒以内に終わらない。32 ビット CAS ループでも 2^16 スレッドで約 2.2 秒。アドレスを分散させれば（1024 アドレス）64 ビット `fetch_add` も 2^20 回で約 0.16 秒 |
| **ワークグループ間の「最後のグループが集計する」パターン**（`lastblock.cpp`） | **誤り**（下記） |

ワークグループ間の集計の誤りについて:

- 各グループが部分和を普通のストアで書き、`atomic_ref<..., acq_rel, device>` でカウンタを増やし、最後のグループが部分和を普通のロードで読む、という構成で試した。
- 16 グループまでは正しい。256 グループでは結果が期待値のおよそ半分（32768 / 65536）になった。
- `atomic_fence(seq_cst, device)` を前後に入れても直らない。
- 部分和の読み書きを `atomic_ref` の load / store（relaxed、device スコープ）にすると、3 回とも正しかった。
- SYCL 2020 のメモリモデルでは、acq_rel の RMW が通常のストアとロードを順序付けるので、この誤りは AdaptiveCpp Metal の不具合と考えられる。
- M3 Ultra は 2 ダイ構成なので、ダイ間でキャッシュの一貫性が保たれていない可能性がある（推測。M3 Max 以下で再現するかは未確認）。

### Kokkos 5.2.0 SYCL バックエンドを acpp でビルド

そのままでは構成できない、またはビルドできない。原因と、今回入れたローカルパッチは次のとおり（全体は `ref/f2_metal/patches/kokkos-5.2.0-acpp-metal.patch`）。

| 問題 | 箇所 | 今回の暫定対処 |
|---|---|---|
| oneDPL が必須扱いになる | `cmake/Modules/FindTPLONEDPL.cmake` | `-DKokkos_ENABLE_ONEDPL=OFF` |
| DPC++ のドライバフラグ `-fsycl` などを付ける。acpp は `-fsycl` を「別実装のフラグ」として致命的エラーにする | `cmake/kokkos_arch.cmake` | `Kokkos_IMPL_SYCL_ACPP=ON` のときは付けない |
| oneAPI 拡張 `sub_group_mask` / `group_ballot` / `this_sub_group`（16 バイト超の lock ベースのアトミックで使う） | desul `Compare_Exchange_SYCL.hpp` など 3 ファイル | ballot を使わない単純なループに変更。16 バイト超でのみ使う経路で、f32 では通らない。GPU の SIMT ではデッドロックの恐れがある |
| `sycl::ext::oneapi::experimental::printf` | `Kokkos_Printf.hpp` | 何もしない関数にする（AdaptiveCpp Metal は printf 非対応） |
| SYCL 2020 の `kernel_bundle` / `get_kernel_id`（AdaptiveCpp に未実装）でワークグループサイズを決めている | ParallelReduce/Scan の Range 版、ParallelFor/Reduce の Team 版 | 固定値（倍数 32、上限 256、sub-group 32） |
| `group_local_memory_for_overwrite`（oneAPI 拡張） | `Kokkos_SYCL_Team.hpp` の team reduce | team reduce 用の scratch を流用（暫定。サイズの検証はしていない） |
| `this_nd_item`（oneAPI 拡張） | `UniqueToken`、`Kokkos_Random.hpp` | 0 番から順にロックを試す（正しいが遅い） |
| `sycl::remquo`、`sycl::nan(uint)` が無い | 数学関数 | 該当関数を除外、`__builtin_nan` |
| **`__SYCL_DEVICE_ONLY__` が定義されない**。AdaptiveCpp の SSCP はホストとデバイスを 1 回のパスでコンパイルするため | `Kokkos_Macros.hpp`、desul `Macros.hpp`（`KOKKOS_IF_ON_DEVICE/HOST`）、`Kokkos_BitManipulation.hpp` | `__acpp_sscp_is_device` / `__acpp_sscp_is_host` で分岐する |

最後の問題は致命的だった。パッチ前は、デバイスで実行されるラムダが「ホスト側」の分岐を通り、View のアクセスチェックが常に失敗していた。

AdaptiveCpp 側のローカルパッチ（`ref/f2_metal/patches/adaptivecpp-metal-num-subgroups.patch`）:

- Metal バックエンドの libkernel に `__acpp_sscp_get_num_subgroups`（`sub_group::get_group_range()`）が実装されていなかった。Kokkos のリダクションがこれを呼ぶと、実行時の MSL コンパイルが失敗する。
- ワークグループのサイズを SIMD 幅で割った値（切り上げ）として実装した。上流に出せる小さな修正。

### パッチ後の Kokkos を Metal GPU で実行

注意: Kokkos は Arch を指定しないと、全デバイスのうち 0 番（AdaptiveCpp の OpenMP ホストデバイス）を選ぶ。最初はこれに気付かず CPU で測っていた。`KOKKOS_DEVICE_ID=1`（または `--kokkos-device-id=1`）で Metal を選ぶ必要がある。`ACPP_VISIBILITY_MASK=metal` では選択が変わらなかった。

| 機能 | scaluq での主な用途 | 結果 |
|---|---|---|
| `parallel_for`（RangePolicy） | ゲート適用全般 | 正しい |
| `parallel_reduce` float / `Kokkos::complex<float>`（RangePolicy） | ノルム、内積、期待値、確率 | **誤り**。256 要素（1 ワークグループ）までは正しく、それを超えると値が欠ける。int では不定値になる |
| `parallel_reduce`（MDRangePolicy） | Operator 期待値、DensityMatrix | **誤り**（同上） |
| TeamPolicy + `TeamThreadRange` の入れ子 reduce | batched 系、3 量子ビット超の密行列ゲート、疎行列ゲート | **誤り**（同上）。チームサイズ `AUTO` だと scratch の上限計算が負になり実行前に失敗するため、明示指定（32 / 256）で確認 |
| `parallel_scan` | サンプリング | **誤り** |
| `Kokkos::Random_XorShift64_Pool` の `frand` | Haar ランダム状態、サンプリング | 動く（平均 0.4965）。パッチにより状態の取得は直列的で遅い |
| `atomic_add(Kokkos::complex<float>)` | `OperatorBatched::apply_to_state` | 正しいが遅い。32 回で約 144 ms、1024 回で約 783 ms（JIT を含む） |

reduce と scan の誤りは、Kokkos が使う「最後のワークグループが集計する 1 パス方式」（`Kokkos_SYCL_ParallelReduce_Range.hpp` など）が、上記の AdaptiveCpp Metal の可視性の不具合に当たっているためと判断した。素の SYCL の再現テストと症状が同じで、ワークグループが 1 個なら正しい。

## scaluq 側で必要になる作業（コード監査、未実装）

upstream/main（`01934038`）の読み取り調査の結果。

- ビルド
  - `SCALUQ_USE_SYCL` は既にあるが、Intel oneAPI（`-fsycl`、`INTEL_GEN`）が前提になっている。acpp 用の分岐が必要。
    - `-fsycl` を付けない。
    - `-fopenmp` を外す。
    - Kokkos の上記オプションを渡す。
  - デバイス選択（`KOKKOS_DEVICE_ID`）を Python から指定できるようにする。
  - f32 だけにするには `-DSCALUQ_FLOAT64=OFF` で足りる。精度ごとにモジュールが分かれる構造はそのまま使える。
- f32 ビルドでも、デバイスカーネル内に double が残る（Metal では JIT 時にコンパイルエラーになる見込み）。
  - `add_state_vector_with_coef` と `multiply_coef`: `StdComplex`（double）をキャプチャして、カーネル内で変換・乗算している（`src/state/state_vector.cpp`、batched 版、DensityMatrix）。
  - `Operator`、`OperatorBatched`、`PauliOperator` の batched 版: `View<Kokkos::complex<double>*>` にカーネルから書いている。
  - `PauliOperator` の `KOKKOS_FUNCTION` の演算子群: `StdComplex` 経由になっている。
  - `get_matrix`: 非 F64 でデバイス上に `StdComplex` の View を確保している（`src/util/utility.cpp`）。
  - 乱数: `drand` / `normal` は double を返す。
  - 修正はいずれも、ホスト側で `Complex<Prec>` に変換してからキャプチャする程度の局所的なもの。CPU 版の性能にも悪影響は無い見込み。
- TeamPolicy を使う箇所が多い。対象は batched 系全般、3 量子ビット超の密行列ゲート、疎行列ゲート、`Operator` の項ごとの期待値、partial trace。単一 `StateVector` の主要経路は、RangePolicy の `parallel_for` / `parallel_reduce` だけで済む。
  - 主要経路の例: 1〜3 量子ビットのゲート、Pauli 回転、ノルム、内積、周辺確率、単一 `PauliOperator` の期待値。
- `Complex<F32>`（8 バイト）の `atomic_add` は `apply_pauli.hpp` の 1 か所だけ。64 ビット CAS になるので、Metal では遅い。
- インデックスはカーネル内で `std::uint64_t`。64 ビット整数演算の正しさは確認した。速度への影響は未測定。

## 性能（参考値、再測定が必要）

測定時、ほかのジョブが GPU を占有していた（`ioreg` の Device Utilization が 94〜100%。自分のプロセスが無い状態でも同じ値）。参考として、MLX でも同じ axpy が 1 回 12 ms、約 20 GB/s しか出なかった。以下は**妥当な性能値ではない**ので、GPU が空いているときに再測定する。

- 素の SYCL の f32 axpy: 2^24 要素で 1 回 6.5〜13 ms（約 15〜31 GB/s）、2^16 要素でも約 1.5 ms/回（ほぼ起動遅延）。
- AdaptiveCpp の `AdaptiveCpp_coarse_grained_events` を有効にすると、2^20 要素で 4.9 ms から 1.8 ms に縮んだ。公式文書が言う「イベントごとのコマンドバッファ」の影響は実際にある。
- 1 ゲート 1 カーネルの scaluq では、起動遅延が性能を大きく左右する。GPU が空いているときの起動遅延（目標は 10 µs 台）を最初に確認すべき。

## 判断と次の手順（提案）

1. **上流への報告**（小さく、効果が大きい）
   - AdaptiveCpp: Metal バックエンドのワークグループ間の可視性の不具合（`lastblock.cpp` が最小再現）と、`__acpp_sscp_get_num_subgroups` の欠落（パッチあり）。
   - Kokkos: SYCL バックエンドの AdaptiveCpp 対応。対象は `__SYCL_DEVICE_ONLY__` 依存、DPC++ フラグ、`kernel_bundle` 依存、oneAPI 拡張の代替。まず issue で方針を相談する。Kokkos 5.2.0 のソースに AdaptiveCpp / hipSYCL への言及は無い。
2. **性能の再測定**: GPU が空いている時間に、素の SYCL で起動遅延と帯域を測る。比較のため、ネイティブ Metal（metal-cpp で同じ axpy）も測る。この数字で F2 と F4 案 B のどちらを本命にするか決める。
3. **F4 案 B の試作**（並行して）
   - 1 量子ビットゲート、ノルム、内積の 3 カーネルだけを Metal（MSL）で書く。
   - scaluq の f32 `StateVector` のメモリを `MTL::Buffer` として共有し、CPU 版と結果・速度を比べる。
   - AdaptiveCpp の Metal interop（`get_native_allocation`）を使えば、F2 経路とも併用できる。
4. **scaluq 側の準備**（どちらの案でも必要）
   - デバイスカーネル内の double の除去。
   - batched 系の TeamPolicy を使う経路は、Metal 版では無効にするか後回しにする。

## 再現手順（要約）

```sh
brew install llvm@21 lld@21
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk
# AdaptiveCpp (develop) — ref/f2_metal/patches/adaptivecpp-metal-num-subgroups.patch を当てる
L=/opt/homebrew/opt/llvm@21
cmake -S AdaptiveCpp -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$L/bin/clang -DCMAKE_CXX_COMPILER=$L/bin/clang++ \
  -DLLVM_DIR=$L/lib/cmake/llvm -DCLANG_EXECUTABLE_PATH=$L/bin/clang++ \
  -DACPP_LLD_PATH=/opt/homebrew/opt/lld@21/bin/ld64.lld \
  -DACPP_COMPILER_FEATURE_PROFILE=full -DWITH_METAL_BACKEND=ON \
  -DMETAL_INCLUDE_DIR=/path/to/metal-cpp -DWITH_OPENCL_BACKEND=OFF \
  -DCMAKE_INSTALL_PREFIX=$PREFIX/acpp
# Kokkos 5.2.0 — ref/f2_metal/patches/kokkos-5.2.0-acpp-metal.patch を当てる
cmake -S kokkos -B build-acpp -DCMAKE_CXX_COMPILER=$PREFIX/acpp/bin/acpp \
  -DCMAKE_CXX_STANDARD=20 -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_ONEDPL=OFF -DKokkos_IMPL_SYCL_ACPP=ON
# 実行時は Metal を明示する
KOKKOS_DEVICE_ID=1 ./kk 2
```

## Web 調査について

Gemini による文献調査（AdaptiveCpp と Kokkos の文書・issue）も行った。

- 採用したもの: AdaptiveCpp の公式文書の記述。ローカルの `doc/install-metal.md` と照合済み。
- 採用しなかったもの: 「Kokkos の Metal バックエンド計画は issue #4953 で却下」「Metal 量子シミュレータ MettleQ」などの記述。#4953 は CUDA の複数アーキテクチャの issue で、MettleQ のリポジトリも存在しなかったため。

Kokkos が AdaptiveCpp を公式にサポートしていないことは、Kokkos 5.2.0 のソースで直接確認した（AdaptiveCpp・hipSYCL・acpp への言及が無い）。
