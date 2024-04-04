# C++ チュートリアル

## プログラムの開始と終了

Qulacs2023の関数やデータ構造は、すべて`qulacs`名前空間に定義されています。

Qulacs2023を利用する際には、使用するリソースを初期化するために、プログラム開始時に必ず`qulacs::initialize()`を呼び出す必要があります。
これをしない場合エラーとなり、プログラムが終了します。

同様に、プログラムの終了の際にはリソースを解放するため、`qulacs::initialize()`を呼び出す必要があります。
`qulacs::initialize()`を呼び出す際には、すべての`qulacs::StateVector`オブジェクトがデストラクトされていることを要請します。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    // 実行するソースコード

    }
    qulacs::finalize();
}
```

## 量子状態

### 量子状態の生成

以下のコードで $n$ qubitの量子状態を生成します。
生成した量子状態ははじめ、 $|0\rangle^{\otimes n}$ に初期化されています。
メモリが不足している場合はプログラムが終了します。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    // 5-qubitの状態を生成
    // |00000>に初期化されている
    const unsigned int n = 5;
    qulacs::StateVector state(n);

    }
    qulacs::finalize();
}
```

### 量子状態の状態ベクトルの取得

量子状態を表す $2^n$ の長さの配列を、`std:vector<qulacs::Complex>`型として取得します。
特にGPUで量子状態を作成したり、大きい $n$ では非常に重い操作になるので注意してください。
配列の要素の型である`qulacs::Complex`型は、ほとんど`std::complex`と同じように扱うことができます。

```cpp
#include <iostream>
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    const unsigned int n = 5;
    qulacs::StateVector state(n);

    // 配列を取得
    auto data = state.amplitudes();

    }
    qulacs::finalize();
}
```

### 量子状態の初期化

生成した量子状態を初期化するために、いくつかの方法が用意されています。
以下のようにすると、計算基底に初期化したり、ランダムな状態に初期化したりできます。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    const unsigned int n = 5;
    qulacs::StateVector state(n);

    // |00000>に初期化
    state.set_zero_state();
    // |00101>に初期化
    state.set_computational_basis(0b00101);
    // ランダムな初期状態を生成（実行毎にランダム）
    state = qulacs::StateVector::Haar_random_state();
    // シードを指定してランダムな初期状態を生成
    state = qulacs::StateVector::Haar_random_state(0);

    }
    qulacs::finalize();
}
```
`std::vector<qulacs::Complex>`を介して初期化することや、
インデクスを指定して単一の要素のみを初期化することもできます。
`std::vector`を介する場合、サイズは$2^n$であることが要請されます。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    const unsigned int n = 5;
    const unsigned int dim = 1 << 5; // (=2^5=32)
    qulacs::StateVector state(n);

    // 量子状態のデータを配列から初期化
    // すべての状態の確率振幅を等しくする例
    std::vector<qulacs::Complex> vec(dim, Complex(1, 0));
    state.load(vec);
    state.normalize(); // Σ(確率振幅)^2=1 となるよう正規化

    // |00000>と|11111>の確率振幅が等しくなるように初期化
    state.set_zero_norm_state();
    state.set_amplitude_at_index(0b00000, Complex(1, 0));
    state.set_amplitude_at_index(0b11111, Complex(1, 0));
    state.normalize();

    }
    qulacs::finalize();
}
```

### 量子状態のコピーとライフタイム

次のようにして量子状態を複製できます。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    const unsigned int n = 5;
    qulacs::StateVector state(n);

    // コピーして新たな量子状態を作成
    auto second_state = state.copy();

    }
    qulacs::finalize();
}
```

`qulacs::StateVector`の代入操作は、状態ベクトルの保存領域を**シャローコピー**します。
次のように`copy`メソッドを介さず代入操作を行った場合、`state`と`second_state`は内部で同じ状態ベクトルを共有して保持することになるため、期待した実行結果となりません。

```cpp
#include <state/state_vector.hpp>

int main(){
    qulacs::initialize();
    {

    const unsigned int n = 5;
    qulacs::StateVector state(n);
    state.set_computational_basis(0b00101);

    // 新たな量子状態を作成できていない
    // stateとsecond_stateは同じ状態ベクトルを共有してしまう
    auto second_state = state;

    }
    qulacs::finalize();
}
```

`qulacs::StateVector`は内部に参照カウンタを持っており、データの管理を`std::shared_ptr`のように行っています。
つまり、ある`qulacs::StateVector`オブジェクトが生成されてから、そのオブジェクトが参照されなくなるまでが、オブジェクトのライフタイムです。

### 量子状態に関する計算

他にもさまざまなメソッドが用意されています。
詳しくはドキュメント（未準備）を参照してください。

```cpp
#include <state/state_vector.hpp>

int main() {
    qulacs::initialize();
    {

    const unsigned int n = 5;
    qulacs::StateVector state(n);
    
    // 2乗ノルムの計算
    double norm = state.compute_squared_norm();
    // Z基底で測定した時のentropyの計算
    double entropy = state.get_entropy();

    // index-th qubitをZ基底で測定して0を得る確率の計算
    unsigned int index = 3;
    double zero_prob = state.get_zero_probability(index);

    // 周辺確率を計算する
    // 以下は0,4-th qubitが0、1,2-th qubitが1と測定される確率の例
    // 0,1以外の値は無効値
    std::vector<unsigned int> vals{ 0,1,1,2,0 };
    double marginal_prob = state.get_marginal_probability(vals);

    }
    qulacs::finalize();
}
```
