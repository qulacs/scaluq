# 上流への報告（2026-11-07 提出）

`filed/` は、提出したときの本文（英語）。

| 提出先 | 番号 | 内容 |
|---|---|---|
| AdaptiveCpp | #2310 | SYCL 2020 のアトミック能力照会が未実装 |
| AdaptiveCpp | #2311 | Metal: 最後のワークグループが集計する reduce が誤る（RMW が relaxed 扱い、フェンス後も他グループの通常ストアが見えない） |
| AdaptiveCpp | #2312 | Metal: `__acpp_sscp_get_num_subgroups` が未実装（修正案付き） |
| AdaptiveCpp | #2313 | Metal: ローカルとグローバルのポインタの実行時選択が変換できない（文書化の依頼） |
| Kokkos | #9750 | SYCL バックエンドの AdaptiveCpp 対応の相談（Discussions が無効なので issue） |
| scaluq | #471 | `Operator::get_expectation_value` が項ごとの計算の 5〜18 倍遅い |
| scaluq | #472 | f32 の `get_squared_norm`/`normalize` の精度劣化 |

- 再現プログラムは `../tests/lastblock.cpp` と `../tests/repro/`。scaluq の 2 件は上流 main（01934038）の OpenMP ビルドで確認した。
- 出さなかった項目: 「イベントがキューより後に破棄されると落ちる」。単独の SYCL プログラム（`../tests/repro/event_outlives_queue.cpp`、`SYCLInternal` と同じ宣言順を含む 3 通り）では落ちなかったため。Kokkos の issue には、単独では再現しないと書いた。
