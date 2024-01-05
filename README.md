# qulacs2023

## 依存ライブラリ
- GCC 11 以上
- CMake 3.21 以上
- CUDA 12.2 以上（GPU利用時のみ）

## ビルド・実行方法

### ビルド (CPU)
```
script/build_gcc.sh
```

### ビルド（GPU）
```
QULACS_USE_CUDA script/build_gcc.sh
```

※キャッシュ変数がセットされるため、ターゲットを変えてビルドする際は `build/CMakeCache.txt` を削除する

### テスト
```
ninja -C build test
```

### qulacs2023 を用いての C++ 単一ファイルの手元実行
`exe` の中に cpp ファイルを作成し、`exe/CMakeLists.txt` に追記してビルド

### フォーマット
```
ninja -C build format
```

### Python へのインストール
要確認
