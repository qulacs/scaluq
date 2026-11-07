# F2 調査報告: Kokkos + AdaptiveCpp + Metal で scaluq を Apple Silicon GPU で動かせるか

調査日: 2026-11-07
対象: `ref/scaluq_improvements.md`（統合ブランチ側）の F2・F3 の手順 1〜2 と、手順 3 の机上評価
結論の扱い: 実機で確認した事実と、コードを読んで推定した事項は分けて書く。性能値は参考値（後述の理由で再測定が必要）。

## 更新（2026-11-07 第 4 版）: GPU が空いた状態で性能を測った

OS を再起動して GPU を空にし、第 3 版で準備した測定（P1・P2）を行った。生データは `ref/f2_metal/bench/results-2026-11-07/` にある。

**要点**

- 状態が大きいとき（n ≥ 24）、scaluq のゲートは**ネイティブ Metal と同じ速さ**になった。H、X、CX、RX はどれも約 700 GB/s で、CPU（OpenMP 28 スレッド）の約 2.6〜3 倍。大きい状態のゲートは、ネイティブ Metal に置き換える必要がない。
- 差が大きいのは**起動遅延**。scaluq は 1 ゲート約 65 µs かかる。ネイティブ Metal でまとめて投入すると 1 カーネル約 1 µs、素の SYCL で `AdaptiveCpp_coarse_grained_events` を有効にすると約 9 µs。n ≤ 20 では、この遅延が実行時間のほとんどを占める。
- 次に差が大きいのは reduce（ノルム）。n = 28 でネイティブの 1.4 倍の時間がかかり、n ≤ 20 では約 0.27 ms の固定コストがかかる。
- `Operator` の期待値は、GPU でも CPU でも同じ状態のノルム計算の 20〜55 倍遅い。Metal の問題ではなく、scaluq 上流の実装の問題（後述）。
- 第 3 版で 606 秒かかった `MergeGate` は、**43.7 秒**になった。前回の遅さは GPU が混雑していたためだった。
- 測定中に 2 つの不具合が見つかった。(1) 大きな状態でデバイス上の Haar ランダム状態を作ると、GPU カーネルが暴走して残る。(2) f32 乱数で NaN が出る。(2) は PR #464 で直した。

### 測定条件

- Apple M3 Ultra（CPU 32 コア）、macOS 26.7.1。起動直後に測り、測定中の GPU 使用率は 0% だった。CPU 台帳で専有枠を予約した。
- scaluq: `metal_bench`（f32、`ExecutionSpace::Default`）。Metal ビルド（AdaptiveCpp、`KOKKOS_DEVICE_ID=1`）と OpenMP ビルド（g++-16、`OMP_NUM_THREADS=28`、`OMP_PROC_BIND=spread`）で、同じプログラムを使った。
- ネイティブ Metal: `native_bench`（MSL を実行時にコンパイル）。`batched` は 1 つのエンコーダに続けて投入し、最後に 1 回待つ。`single_cb` は 1 回ごとにコマンドバッファを確定して待つ。
- 時間は 1 回あたりのウォールクロック時間（ms）。GB/s は H なら状態の読み書き（状態サイズ × 2）を時間で割った値。

### 起動遅延

| 測り方 | 1 回あたり |
|---|---|
| ネイティブ Metal、空カーネル、1 コマンドバッファごとに確定して待つ | 0.17〜0.25 ms（GPU 上の実行時間は 4 µs） |
| ネイティブ Metal、空カーネル、まとめて投入 | 1.0〜1.4 µs |
| 素の SYCL、triad 2^16 要素（既定） | 0.152 ms |
| 素の SYCL、triad 2^16 要素（`AdaptiveCpp_coarse_grained_events`） | 0.009 ms |
| scaluq（Metal）、1 ゲート、n = 16 | 0.060〜0.073 ms |
| scaluq（OpenMP 28 スレッド）、1 ゲート、n = 16 | 0.13〜0.18 ms |

- scaluq は 1 ゲートごとにホストで待つわけではないので、ネイティブの「1 回ごとに待つ」より速い。ただし、まとめて投入した場合の約 50 倍かかる。
- 素の SYCL では、`coarse_grained_events` で 1 回 0.152 ms が 0.009 ms になった。Kokkos の SYCL バックエンドが作るキューにこの性質を付けられれば、scaluq の起動遅延も大きく縮む見込みがある（未確認）。

### 状態ベクトルの操作（ms/回）

| 操作 | n | ネイティブ Metal | scaluq Metal | scaluq CPU 28 スレッド |
|---|---|---|---|---|
| H（t = 0） | 20 | 0.029 | 0.068 | 0.217 |
| H（t = 0） | 24 | 0.389 | 0.384 | 1.135 |
| H（t = 0） | 28 | 6.16 | 6.15 | 16.2 |
| CX（制御 0、標的 n−1） | 24 | 0.370 | 0.373 | 0.790 |
| CX（制御 0、標的 n−1） | 28 | 6.13 | 6.13 | 11.3 |
| ノルム | 20 | 0.025（1 回ごとに待つと 0.227） | 0.273 | 0.195 |
| ノルム | 24 | 0.201（同 0.413） | 0.556 | 0.897 |
| ノルム | 28 | 3.12（同 3.34） | 4.50 | 10.9 |
| 期待値（2 項） | 24 | — | 15.0 | 17.9 |
| 期待値（2 項） | 28 | — | 250 | 287 |

- ゲート: n ≥ 24 では、scaluq Metal とネイティブ Metal の差は 1% 以内。約 700 GB/s は M3 Ultra の公称帯域（819 GB/s）の約 85%。
  - triad（4 バイトの float 単位で読み書き）は約 333 GB/s で、H の半分しか出ない。8 バイト単位でアクセスするかどうかの違いと推定しているが、確認していない。
- ノルム: scaluq Metal は、ネイティブで「1 回ごとに待つ」場合より遅い。2 パスの reduce と結果のホストへの読み戻しで、約 0.27 ms の固定コストがかかっている。
- 期待値: 項が 2 つしかないのに、ノルムの 20〜55 倍かかる。GPU と CPU で同じ傾向なので、実装の問題と考える（後述）。

### 測定中に見つかった問題

1. **デバイス上の Haar ランダム状態で GPU カーネルが暴走して残る**
   - n = 24 で `StateVector::Haar_random_state` を Metal 上で呼ぶと、10 分以上終わらなかった。
   - プロセスを終了させても、GPU 使用率は 100% のまま下がらなかった。空カーネルの起動は 2.3 ms に悪化し、OS を再起動するまで戻らなかった。
   - 第 3 版で「ほかの処理で使用率 100%」と書いた状態も、同じ原因だったと考える。以前の実行で残ったカーネルが GPU を使い続けていた。
   - 原因（推定）: `Random_XorShift64_Pool` の `get_state()` は、状態のロックをスピンで取り合う。要素数（2^24）がプールの状態数よりずっと多いと、多くのスレッドが同じロックを待つ。Metal は、待っているスレッドとロックを持つスレッドが並行して進むことを保証しないので、事実上止まる。
   - 対処: `metal_bench` は、Haar 状態をホスト（HostSerial）で作って `load()` するように変えた。既存テストの Haar 状態は小さい（n ≤ 10 程度）ので、問題は出ていない。
   - 本格的に直すには、Kokkos パッチの乱数状態の割り当てを、ロックを取り合わない方式（スレッド番号から状態を直接決めるなど）にする必要がある。

2. **f32 乱数で NaN が出る**（PR #464 で修正）
   - Kokkos の `frand()` は、64 ビット整数を float で 2^64 で割る。丸めで、ちょうど 1.0f が返ることがある（約 2^-25 の確率）。
   - `SCALUQ_NO_DEVICE_DOUBLE` のときの Box–Muller は `log(1 - frand())` を計算するので、1.0f が出ると `log(0)` で無限大になる。n = 23 以上（2^24 回以上引く）では、ほぼ確実に NaN の振幅が混ざっていた。
   - サンプリングの `uniform01` も、[0, 1) のはずが 1.0 を返し得た。
   - 修正: `urand()` の上位 24 ビットから [0, 1) の float を作る関数 `uniform01_float` を追加し、両方で使うようにした。`SCALUQ_NO_DEVICE_DOUBLE` のビルドだけが対象で、ほかのビルドの乱数列は変わらない。
   - Metal 上の乱数・サンプリング・`MergeGate` のテスト 39 件が成功した（`tests_random_merge_gpu.txt`）。f32 の既存テスト全体は、この修正のあと流し直していない。

3. **f32 の HostSerial と OpenMP で、正規化の精度が悪い**（scaluq 上流の既存の問題、未修正）
   - HostSerial の f32 では、ノルムを float で順に足していく。n = 24 の Haar 状態を正規化しても、double で計算したノルムは 1.024 だった。OpenMP でも、n = 28 で 0.999。
   - Metal（ツリー型の reduce）では、この誤差は出ない。

4. **`Operator::get_expectation_value` が遅い**（scaluq 上流の既存の問題、未修正）
   - `MDRangePolicy<Rank<2>>` で、{項数, 2^(n−1)} の 2 次元 reduce をしている。項が少ないと極端に細長い形になり、タイルの分割が合っていないと推定している。
   - 2 項、n = 28 で、Metal 250 ms、CPU 287 ms。同じ状態のノルム（4.5 ms、10.9 ms）の 26〜55 倍。
   - Metal 固有の問題ではない。上流で、項ごとに 1 次元の reduce をするなどの改善を検討するのがよい。

### 判断（どこをネイティブ Metal にするか）

| 対象 | 判断 |
|---|---|
| 大きい状態のゲート | 置き換え不要。F2 の経路ですでにネイティブ並み |
| 起動遅延（小さい状態、ゲートの多い回路） | まず Kokkos の SYCL キューに `coarse_grained_events` を付けて試す。素の SYCL では 17 倍速くなった。足りなければ、ゲート列をまとめて 1 つのコマンドバッファに入れる方式をネイティブで検討する |
| reduce（ノルム、内積） | 次の候補。大きい状態で 1.4 倍、小さい状態では固定コストが大きい。Kokkos の 2 パス reduce の読み戻しを減らせるか先に調べる |
| 期待値 | ネイティブ化ではなく、上流のアルゴリズムを直す（CPU でも遅い） |

### 次の作業（第 4 版時点）

| 優先度 | 作業 |
|---|---|
| P1 | Kokkos の SYCL キューで `AdaptiveCpp_coarse_grained_events` を試し、scaluq の 1 ゲートの時間を測り直す |
| P1 | 乱数状態の割り当てを、ロックを取り合わない方式にする（暴走カーネルの根本対策） |
| P2 | f32 の既存テスト全体（229 件）を、乱数の修正のあとで流し直す |
| P2 | reduce の固定コストの内訳を調べる |
| 上流 | scaluq に issue を出した: #471（`get_expectation_value` の性能）、#472（f32 の正規化精度）。どちらも上流 main の OpenMP ビルドで再現を確認した |
| 上流 | AdaptiveCpp に #2310〜#2313、Kokkos に #9750 を出した（乱数プールの件と性能結果を含む）。一覧は `ref/f2_metal/upstream/README.md`。返答を見て、Kokkos パッチの分割と PR 化を進める |

## 更新（2026-11-07 第 3 版）: f32 の既存テストが Metal GPU で 229/229 成功

第 2 版の「次の作業」を優先度順に進めた。**f32 の既存テスト 229 件がすべて Metal GPU 上で成功した。** OpenMP（f32/f64）の既存テスト 968 件もすべて成功し、既存の CPU ビルドを壊していないことを確認した。性能の測定はまだしていない（後述）。

| 優先度 | 作業 | 結果 |
|---|---|---|
| P0 | デバイス内 double の除去（残り） | **完了**。17 件が成功に変わった |
| P1 | 終了時の SIGSEGV | **原因を特定して回避**。22 件が正常終了するようになった |
| P3 | `parallel_scan` の 2 パス化 | **完了**。サンプリング 4 件が成功に変わった |
| P3 | 乱数の再現性（シード） | **完了**。`HaarRandomStateSameSeed` 2 件が成功に変わった |
| — | threadgroup/device ポインタの `select` | **scaluq 側で回避**。2 件が成功に変わった |
| P1 / P2 | 起動遅延・帯域の再測定、ネイティブ Metal との比較 | 測定プログラムは用意した。GPU がほかの処理で使用率 100% のため、測定は未実施 |

### scaluq 側の変更（`exp/f2-metal-scaluq`、PR #464 の 2 コミット目）

- Operator 系の double の除去:
  - `Operator`、`OperatorBatched`、`PauliOperator` の batched 版の期待値・遷移振幅は、`View<Kokkos::complex<double>>` にデバイスから書いていた。作業精度の `View<ComplexType>` に書き、ホストで `StdComplex` に広げるように変えた（共通関数 `internal::complex_view_to_std_vector`）。
  - `convert_internal_matrix_to_external_matrix`（`get_matrix` が使う）も、ホストにコピーしてから double に広げるようにした。
  - `PauliOperator` に、デバイス用の既定コンストラクタと、作業精度の係数を受け取る内部コンストラクタを追加した。`get_dagger`、`operator*`、`Operator::operator*=` は、デバイス上で `StdComplex`（double）を経由しなくなった。
- threadgroup/device ポインタの `select`:
  - 原因は `multi_dense_matrix_gate_scalar` だった。スクラッチのレベル（0 = threadgroup、1 = device）を実行時に選んでいるため、2 種類のアドレス空間のポインタを `select` で選ぶ IR になる。Metal には汎用アドレス空間が無いので、MSL に変換できない。
  - `SCALUQ_SYCL_ACPP` のときだけ、レベルをコンパイル時定数（`std::integral_constant`）にした。ほかのビルドは従来どおり `int` を渡すので、生成されるコードは変わらない。
- 変更量は 7 ファイル、+134/−90 行。

### Kokkos パッチの追加（`ref/f2_metal/patches/kokkos-5.2.0-acpp-metal.patch`）

- **終了時の SIGSEGV**:
  - 原因: `SYCLInternal` では `m_queue` が `m_team_scratch_event[]` より後に宣言されている。そのため、デストラクタはキューを先に破棄し、イベントを後で破棄する。AdaptiveCpp は、in-order キューの破棄後にそのイベント（`dag_node`）を破棄すると落ちる。
  - 調べ方: デバッガは権限が無く使えなかったため、SIGSEGV ハンドラを `DYLD_INSERT_LIBRARIES` で注入して、バックトレースを取った（`Kokkos::finalize` → `~SYCLInternal` → `hipsycl::rt::dag_node::~dag_node`）。
  - 回避: デストラクタ内で、キューがまだ有効なうちにイベントを解放する（1 行）。
  - SYCL では、イベントがキューより長く生きてもよいはず。したがって、本来は AdaptiveCpp 側の不具合と考える。
- **`parallel_scan` の 2 パス化**:
  - 1 パス目では、「最後のワークグループがグループ合計を走査する」処理を、ワークグループが 1 個のときだけ使う。
  - ワークグループが複数あれば、グループ合計の走査を単一ワークグループのカーネルとして投入し、そのあと既存の結果更新カーネルを流す。
  - 新しいテスト `scan.cpp`（64 ビット整数と float、n = 32〜2^24、2 のべき乗でない長さを含む、10 回ずつ）で 90/90 正しい。
- **乱数の再現性**:
  - `Random_UniqueIndex` は、乱数状態の割り当てを、ほかのバックエンドと同じくグローバルなスレッド番号から探すようにした。スレッド番号は AdaptiveCpp の SSCP 組み込み関数（`__acpp_sscp_get_local_id_x` など）から計算する。
  - 新しいテスト `rnd.cpp` で、プールの状態数以下の並列度では同じシードで同じ乱数列になることを確認した（n ≤ 65536）。
  - n = 2^20 のように状態数を超えると、乱数列は一致しない。複数のスレッドが同じ状態を取り合うためで、CUDA など他のバックエンドでも同じ性質。
- `UniqueToken` は、0 番から空きを探すまま。scaluq は使っていない。
- Kokkos への変更は、合計 22 ファイル、+306 行になった。

### Metal 上のテスト結果（第 3 版）

| 結果 | 第 2 版 | 第 3 版 |
|---|---|---|
| 成功 | 182 | **229** |
| 成功（終了時 SIGSEGV） | 22 | 0 |
| double で JIT 不可 | 17 | 0 |
| コード生成エラー | 2 | 0 |
| 失敗 | 2 | 0 |
| タイムアウト | 4 | 0 |

- 注意: `MergeGateTest` は成功するが、1 件で 606 秒かかった（300 秒の打ち切りでは TIMEOUT になる）。
  - 測定中、GPU はほかの処理で使用率 100% だった。小さなカーネルの起動と同期を多数繰り返すテストなので、待ち時間が積み上がったと推定している。
  - 性能測定のときに、あわせて確認する。
- 全件の結果は `ref/f2_metal/scaluq_f32_metal_tests.tsv`。

### 性能測定の準備（P1・P2）

- `ref/f2_metal/bench/metal_bench.cpp`: scaluq の公開 API で f32 の H、X、CX、RX、ノルム、期待値の時間と実効帯域を測る。
  - Metal ビルドと OpenMP ビルドの両方で、同じプログラムを使う。
  - `exe/` に置き、`exe(metal_bench)` を追加してビルドする。PR には含めていない。
- `ref/f2_metal/native/`: ネイティブ Metal の比較用プログラム（Objective-C++、MSL は実行時にコンパイル）。
  - 測る項目: 空カーネルの起動遅延（1 コマンドバッファ 1 回と、まとめて投入の 2 通り）、triad 帯域、H、CNOT、2 パスのノルム。
  - 小さいサイズでの正しさの確認（H と CNOT を CPU 参照と比較、ノルムの相対誤差 1e-7 程度）は通った。
- GPU の使用率が 100% のときの値は使わない。GPU が空いている時間に、Metal ビルド・OpenMP ビルド・ネイティブ Metal の 3 つを続けて測る。

### その他（上流 scaluq の既存コードで見つけた点）

- `OperatorBatched::operator*(const std::vector<StdComplex>&)` と `operator*=`:
  - 係数の配列はバッチ数の長さなのに、ループは全項（`_ops` の要素数）を回り、項の番号で配列を引いている。項の数がバッチ数より多いと、範囲外を読む。
  - テストが無く、今回の変更とは無関係なので直していない。別の issue にするのがよい。

### 次の作業（第 3 版時点）

| 優先度 | 作業 | 状態 |
|---|---|---|
| P1 / P2 | 起動遅延・帯域、scaluq の各操作、ネイティブ Metal の比較測定 | 準備済み。GPU が空くのを待つ |
| P2 | 遅い部分だけネイティブ Metal に置き換える | 測定結果を見て判断 |
| P3 | TeamPolicy / batched の性能 | 未着手 |
| 上流 | AdaptiveCpp への報告 | 下書き予定。RMW の memory_order 無視、`__acpp_sscp_get_num_subgroups` の欠落、キュー破棄後のイベント破棄で落ちる件、capabilities クエリの未実装、汎用アドレス空間の `select` が変換できない件（制約として明記を依頼） |
| 上流 | Kokkos への相談 | 下書き予定。AdaptiveCpp 対応一式、2 パス reduce/scan、`~SYCLInternal` の破棄順 |

## 更新（2026-11-07 第 2 版）: 2 パス reduce で reduce が正しくなった

レビューを受けて判断を改めた。**F2 は継続する。** 基本ゲートは Kokkos の `parallel_for` のまま使い、Metal で問題になる処理だけを個別に直す。

- Kokkos の SYCL リダクションを、AdaptiveCpp 向けだけ 2 パス方式に変更した。
  - 1 パス目: 各ワークグループは部分和を書くだけにする。「最後のグループが集計する」処理はワークグループが 1 個のときだけ使う。
  - 2 パス目: ワークグループが複数あれば、単一ワークグループの集計カーネルを同じ in-order キューに投入する。
- 対象は Range・MDRange・TeamPolicy の 3 つ。共通ヘルパー `two_pass_final_reduction` にまとめ、約 100 行を追加した（`ref/f2_metal/patches/kokkos-5.2.0-acpp-metal.patch` に含む）。
- Metal GPU 上の結果:

| テスト | 第 1 版 | 2 パス化後 |
|---|---|---|
| `parallel_reduce` float / int（n = 32〜2^20、`red.cpp`） | 257 以上で誤り | すべて正しい |
| `parallel_reduce` `complex<float>` の二乗和と、float の `norm` の和（n = 2^8〜2^26 × 20 回 = 160 ケース、`red2.cpp`。CPU の倍精度の参照値と比較） | — | **160/160 正しい**（相対誤差は最大 1.2e-6） |
| MDRange reduce（1024×1024） | 誤り | 正しい |
| TeamPolicy + `TeamThreadRange` の入れ子 reduce（チームサイズ 32/64/128/256） | 誤り | 正しい |
| `parallel_scan` | 誤り | 誤り（未対応。複数パス化は P3） |

- 2 パス化はとりあえずの回避策というより、Metal に合った方式と考える。理由は次の 2 つ（下記「原因の追加調査」も参照）。
  - AdaptiveCpp の Metal バックエンドは、RMW アトミックの memory_order を常に relaxed で出力している。
  - 明示的な device スコープのフェンスを入れても、通常のストアがほかのスレッドグループから見えない。
- 2 パス化で、1 回の reduce につきカーネル起動が 1 回増える。性能への影響は、GPU が空いているときに測る（P1）。

### 原因の追加調査（レビューの A 点）

- AdaptiveCpp は SYCL 2020 のクエリ `atomic_memory_order_capabilities`、`atomic_memory_scope_capabilities`、`atomic_fence_order_capabilities`、`atomic_fence_scope_capabilities` を実装していない。コンパイルエラーになるため、デバイスが報告する対応範囲は確認できなかった。
- Metal 用 libkernel（`src/libkernel/sscp/metal/atomic.cpp`）の実装:
  - load と store は、relaxed のアトミック操作の前後に、order に応じた `atomic_thread_fence` を入れている。
  - **`fetch_add` などの RMW（23 種類）は、order とスコープの引数を無視して常に `memory_order_relaxed` で出力し、フェンスも入れていない**。Kokkos が使う `atomic_ref<..., acq_rel, device>` の `++` は、実際には relaxed になる。
  - これは AdaptiveCpp に報告すべき不具合。
- 一方、明示的な `atomic_fence(seq_cst, device)` は、`atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` に変換されている（`barrier.cpp`）。それでも、ほかのグループの通常ストアは見えなかった。アトミックの load/store で読み書きした場合だけ正しかった。
  - 推定: Metal では、アトミックでない device メモリは、同じディスパッチ内のスレッドグループ間で一貫性を持たない。保証されるのはカーネルの境界か、アトミック操作を使った場合だけと考えられる。MSL 3.2 の `coherent(device)` 修飾との関係は、MSL 仕様書で要確認。
  - したがって、RMW を直しても 1 パス方式が正しくなるとは限らない。2 パス方式が安全側。
- グループ内のバリアは、木構造リダクションと `reduce_over_group` の結果が正しいことから、正しく動いていると判断した。
- M1 や M3 Max など 1 ダイのチップで再現するかは、手元に機材が無いため未確認。2 ダイ構成が原因かどうかは断定しない。

### scaluq f32 版を Metal GPU で実行（P0・P1 の途中経過）

- 作業ブランチは `exp/f2-metal-scaluq`（upstream/main 起点）。scaluq 側の変更は次の 2 つ（約 40 行）。
  - CMake: `SCALUQ_SYCL_ACPP=ON` のとき、`-fsycl` を付けない。同時に `SCALUQ_NO_DEVICE_DOUBLE` を定義し、Kokkos に `Kokkos_IMPL_SYCL_ACPP=ON` と `Kokkos_ENABLE_ONEDPL=OFF` を渡す。
  - デバイス内の double の除去（一部）
    - 乱数: `util/device_random.hpp` を追加。`SCALUQ_NO_DEVICE_DOUBLE` のときだけ、f32 以下は `frand` と float の Box–Muller を使う。それ以外のビルドは従来どおり `drand`/`normal`（double）で、シード付きの結果は変わらない。
    - 係数: `add_state_vector_with_coef` と `multiply_coef` で、係数をホスト側で `Complex<Prec>` に変換してからキャプチャする。対象は StateVector、StateVectorBatched、DensityMatrix。
- ビルド: `-DSCALUQ_USE_OMP=OFF -DSCALUQ_USE_SYCL=ON -DSCALUQ_SYCL_ACPP=ON -DSCALUQ_FLOAT64=OFF -DFETCHCONTENT_SOURCE_DIR_KOKKOS=<パッチ済み Kokkos>`。C++ テストは acpp でビルドできた（Python バインディングは未試行）。
- テストの実行方法: `KOKKOS_DEVICE_ID=1` を指定し、`F32DefaultSpace`（= Metal GPU）の 229 件を 1 件ずつ実行した。1 件ずつにしたのは、1 件の JIT エラーでプロセスごと落ちるため。全件の結果は `ref/f2_metal/scaluq_f32_metal_tests.tsv`。

| 結果 | 件数 | 内容 |
|---|---|---|
| 成功 | 182 | |
| 成功（ただし終了時に SIGSEGV） | 22 | テスト本体は `[ OK ]`。プロセス終了時に落ちる。batched 系、疎行列ゲート、`Operator::apply_to_state` など、TeamPolicy や atomic を使うテストに多い。原因は未調査（Kokkos の終了処理と Metal の資源解放の順序を疑っている） |
| デバイス内 double で JIT 不可 | 17 | `Operator` / `OperatorBatched` / `PauliOperator` の batched 版の期待値・遷移振幅（`View<Kokkos::complex<double>>` にカーネルから書く）、`Operator` の Hermitian 判定、`get_matrix`（`utility.cpp` の `StdComplex` View）。監査で挙がっていた箇所のとおり |
| Metal のコード生成エラー | 2 | `MergeGate`、`GeneralDenseMatrix`（4 量子ビット以上の密行列ゲート）。threadgroup と device のポインタを `select` で選ぶ IR を MSL に変換できない（AdaptiveCpp の制約） |
| 失敗 | 2 | `HaarRandomStateSameSeed`。同じシードで同じ状態にならない。Kokkos パッチの乱数状態の割り当て（0 番から順に空きを探す）が非決定的なため。直すにはスレッド番号を得る手段が必要 |
| タイムアウト（300 秒） | 4 | サンプリング。単一状態の `parallel_scan` が未対応なので累積確率が誤る（P3） |

テストスイート別（成功と終了時 SIGSEGV を成功に含める）:

| スイート | 成功/全体 | スイート | 成功/全体 |
|---|---|---|---|
| GateTest | 35/35 | StateVectorTest | 13/16 |
| BatchedGateTest | 27/27 | StateVectorBatchedTest | 11/11 |
| DMGateTest | 31/31 | DensityMatrixTest | 13/15 |
| ParamGateTest | 9/9 | PauliOperatorTest | 9/10 |
| CircuitTest | 13/13 | OperatorTest | 11/20 |
| ParamCircuitTest | 5/5 | OperatorBatchedTest | 6/13 |
| CircuitOptimizeTest | 2/2 | Matrix8x8Test | 1/2 |
| CircuitExpectationGradientTest | 3/3 | GeneralDenseMatrixTest / MergeGateTest | 0/1、0/1 |
| Qasm2Test | 15/15 | | |

**レビューの P1 の完了条件（H・X・CX・回転ゲート、ノルム・内積・期待値が CPU の参照値と一致）は、既存テストの範囲で満たした。** 単一状態のゲートは GateTest で、単一 `PauliOperator` / `Operator` の期待値はそれぞれのテストで確認した。

テストの所要時間は、初回 JIT を含めて 1 件あたり数秒〜数十秒（2 回目以降は JIT キャッシュが効く）。性能の評価には使っていない。

### 次の作業（第 2 版時点）

| 優先度 | 作業 | 状態 |
|---|---|---|
| P0 | 2 パス `parallel_reduce` | **完了**（Range・MDRange・Team） |
| P0 | デバイス内 double の除去 | 乱数・係数は完了。残りは Operator 系の batched の期待値・遷移振幅、Hermitian 判定、`get_matrix` |
| P1 | f32 単一 StateVector の統合 | **既存テストで確認済み**（上表） |
| P1 | 起動遅延と帯域の再測定 | 未実施。GPU が空いている時間が必要 |
| P1 | 終了時の SIGSEGV | 未調査 |
| P2 | ネイティブ Metal との性能比較 | 未実施 |
| P3 | `parallel_scan` の複数パス化、サンプリング | 未実施 |
| P3 | 乱数の再現性（シード） | 未実施。Kokkos の `Random_UniqueIndex` の代替が必要 |
| 上流 | AdaptiveCpp への報告 | 未提出。RMW の memory_order 無視、`__acpp_sscp_get_num_subgroups` の欠落、threadgroup/device ポインタの `select`、SYCL 2020 の capabilities クエリの未実装 |
| 上流 | Kokkos への相談 | 未提出。AdaptiveCpp 対応（`__SYCL_DEVICE_ONLY__`、DPC++ フラグ、`kernel_bundle`、oneAPI 拡張）と 2 パス reduce |

## 結論（第 1 版。上の更新を優先）

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

> 第 4 版で、GPU が空いた状態で測り直した。この節の値は混雑した GPU でのもので、使わない。

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
