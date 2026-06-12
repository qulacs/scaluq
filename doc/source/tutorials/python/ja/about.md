# Scaluqについて

Scaluq は、量子回路シミュレータ [Qulacs](https://github.com/qulacs/qulacs) をもとに再開発された、新しい Python/C++ ライブラリです。  
大規模な量子回路、ノイズを伴う量子回路、パラメトリック量子回路の高速シミュレーションを実行することができます。  
本ライブラリは、MITライセンスの下で公開されています。  

## 特徴

[Qulacs](https://github.com/qulacs/qulacs) に比べ、以下の点が改善されています。

- [Kokkos](https://github.com/kokkos/kokkos) をベースとした実装により、実行環境(CPU/GPU) の切り替えを容易に行うことができます。切り替えの際にコードを変更する必要はありません。
- よりよい実行速度を実現します。
- ポインタをユーザから隠蔽したことにより、より安全に、簡単に記述できます。
- [nanobind](https://github.com/wjakob/nanobind) の導入により、よりコンパクトかつ高速な Python へのバインディングを実現します。
- 複数の量子状態に対して、同じ構造を持ちパラメータのみが異なる量子回路を一括実行するためのバッチ実行機能を提供します。

## パフォーマンス

量子回路シミュレーションの実行時間を、複数の既存量子回路シミュレータと比較しました。  
本ベンチマークでは、CX、RX、RZゲートをターゲット量子ビットを変えながら順に適用する回路を実行し、その平均実行時間を測定しました。

[ベンチマークのリポジトリ](https://github.com/Qulacs-Osaka/benchmark-scaluq) をご確認ください。

### 単一状態ベクトル更新 (2026年1月)

| CPU 結果 | GPU 結果 |
| ------- | --------|
| ![Single State Vector Update (CPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/multiple-gate/multithread/image/circuit.png) | ![Single State Vector Update (GPU)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/multiple-gate/gpu/image/circuit.png) |

### バッチ状態ベクトル更新 (2026年5月)

| バッチサイズを変化させた場合 (#qubits=16) | 量子ビット数を変化させた場合 (batch size=100) |
| -------------------------------------- | ------------------------------------------- |
| ![Batched State Vector Update (batch sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/batch/image/batch_sweep.png) | ![Batched State Vector Update (qubits sweep)](https://github.com/Qulacs-Osaka/benchmark-scaluq/raw/main/benchmark/batch/image/qubits_sweep.png) |

## ビルド時要件

- Ninja 1.10 以上
- GCC 13 以上 または LLVM Clang 13 以上
  - CUDA 利用時はGCC 11以上が利用できるが、Clangは利用不可
- CMake 3.24 以上
- CUDA 12.8 以上（CUDA利用時のみ）
- Python 3.10 以上 (Python利用時のみ)

CUDA を利用する場合は、使用する CUDA がサポートするホストコンパイラのバージョンを使用してください（CUDA Installation Guide の Host Compiler Support Policy を参照）。

※これより低いバージョンでも動作する可能性はありますが確認していません

## 実行時要件

- CUDA 12.8 以上（CUDA利用時のみ）

※これより低いバージョンでも動作する可能性はありますが確認していません
