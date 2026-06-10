# StateVector

量子状態の状態ベクトルは {class}`StateVector <scaluq.default.f64.StateVector>` で表されます。
このクラスは $2^n$ 個の複素数をもちます。ここで、 $n$ は量子ビット数です。

量子ビットのインデックスは $0,1,\dots,n-1$ と番号付けされています。
状態は計算基底によって表されています。例えば、 `[a, b, c, d]` は $a\ket{00}+b\ket{01}+c\ket{10}+d\ket{11}$ を意味します。
基底を下から数えて($0$ からの数え上げ)、 $i$ 番目のビットは量子ビット $i$ です。
例えば、 $\ket{110}$ は量子ビット $0$ が $\ket{0}$ を、量子ビット $1$ が $\ket{1}$ 、そして量子ビット $2$ が $\ket{2}$ を意味します。


{class}`StateVector <scaluq.default.f64.StateVector>` は量子ビット数を意味する`n_qubits`の値によって構築されています。

状態は $\ket{0\dots 0}$ で初期化されます。

```py
from scaluq.default.f64 import StateVector
state = StateVector(2)
print(state)
```
```
Qubit Count : 2
Dimension : 4
State vector : 
  00 : (1,0)
  01 : (0,0)
  10 : (0,0)
  11 : (0,0)
```

## StateVectorのプロパティ取得

{func}`n_qubits <scaluq.default.f64.StateVector>` は量子ビット数 $n$ を返します。

{func}`dim <scaluq.default.f64.StateVector>` はベクトルの次元 $2^n$ を返します。

{func}`get_amplitudes <scaluq.default.f64.StateVector>` はStateVectorの確率振幅を `list[complex]` として返します。

{func}`get_squared_norm()` は、状態のノルムの二乗 $\braket{\phi, \phi}$ を返します。状態が規格化されている場合、この値は $1$ になります。

## StateVectorの初期化

コンストラクタに加えて、関数を用いた状態の初期化が可能です。

{func}`Haar_random_state <scaluq.default.f64.StateVector>` は、状態をHaar random stateに初期化します。オプションとして引数にシード値を渡すことができます。シード値を渡さない場合は、システムのランダムデバイスが使われます。

{func}`uninitialized_state <scaluq.default.f64.StateVector>` は、実行スペース上で初期化を行わずにメモリ確保のみを行います。内容は定義されません。また規格化されていることを保証しません。他の初期化関数よりもこの関数によるベクトルのメモリ確保の方が高速であるため、他のベクトルを後でロードする場合はこの関数を使うことが推奨されます。

```py
from scaluq.default.f64 import StateVector
state = StateVector.Haar_random_state(2)
print("Haar random (without seed): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2)
print("Haar random (without seed): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2, 0)
print("Haar random (seed = 0): ", state.get_amplitudes())
state = StateVector.Haar_random_state(2, 0)
print("Haar random (seed = 0): ", state.get_amplitudes())
state = StateVector.uninitialized_state(2) # 内容は定義されません
```
```
Haar random (without seed):  [(-0.12377148096652951+0.027715511836463032j), (0.23343008413304153-0.6125930779810899j), (0.348536530607012+0.16314293047597564j), (-0.6344277255462337-0.059671766025997705j)]
Haar random (without seed):  [(0.20861863181020227+0.36023007097415805j), (-0.7038208261050227+0.15424679536389918j), (0.032557696049571434+0.4498796978221459j), (0.3196074684536188+0.04422729148920198j)]
Haar random (seed = 0):  [(0.24602695668676106-0.3593147366777609j), (-0.2016366688947537+0.10904346570777179j), (-0.7078115548871466+0.3479734076173536j), (0.09795534521513291-0.3551589695281517j)]
Haar random (seed = 0):  [(0.24602695668676106-0.3593147366777609j), (-0.2016366688947537+0.10904346570777179j), (-0.7078115548871466+0.3479734076173536j), (0.09795534521513291-0.3551589695281517j)]
```

## StateVectorのロード

関数によって、 {class}`StateVector <scaluq.default.f64.StateVector>` の状態ベクトルをロード可能です。

{func}`set_zero_state <scaluq.default.f64.StateVector.set_zero_state>` は、状態ベクトルを $\ket{00\dots0}=[1,0,\dots,0]$ と設定することに使えます。

{func}`set_zero_norm_state <scaluq.default.f64.StateVector.set_zero_norm_state>` は、状態ベクトルを $0=[0,0,\dots,0]$ と設定することに使えます。

{func}`set_computational_basis <scaluq.default.f64.StateVector.set_computational_basis>` は、状態ベクトルを引数 $0\leq b \leq 2^{n}-1$ によって、 $\ket{b}$ と設定することに使えます。

{func}`load <scaluq.default.f64.StateVector.load>` は、 $2^n$ の長さの他のベクトルから直接、確率振幅をロードするために使えます。

```py
from scaluq.default.f64 import StateVector
state = StateVector(2)
state.set_zero_state()
print("zero state:", state.get_amplitudes())
state.set_zero_norm_state()
print("zero norm state:", state.get_amplitudes())
state.load([0.5, 0.5, -0.5, 0.5])
print("loaded state:", state.get_amplitudes())
import numpy as np
state.load(np.array([1, 0, 0, 1]) / np.sqrt(2)) # numpy arraysもロード可能です
print("loaded numpy state:", state.get_amplitudes())
```
```
zero state: [(1+0j), 0j, 0j, 0j]
zero norm state: [0j, 0j, 0j, 0j]
loaded state: [(0.5+0j), (0.5+0j), (-0.5+0j), (0.5+0j)]
loaded numpy state: [(0.7071067811865475+0j), 0j, 0j, (0.7071067811865475+0j)]
```


## StateVectorBatched使用方法

{class}`StateVectorBatched <scaluq.default.f64.StateVectorBatched>` を使う場合、単一の状態ベクトルを2つの方法で取得できます。

- {func}`get_state_vector_at <scaluq.default.f64.StateVectorBatched.get_state_vector_at>` は、コピーを返します。
- {func}`view_state_vector_at <scaluq.default.f64.StateVectorBatched.view_state_vector_at>` は、共有メモリ上のviewを返します。

`view_state_vector_at` は確率振幅をコピーしません。そのため、返された {class}`StateVector <scaluq.default.f64.StateVector>` を変更すると、それに対応する元の `StateVectorBatched` も変更されます。

```py
from scaluq.default.f64 import StateVectorBatched

states = StateVectorBatched(2, 2)

# View: in-placeでbatch 0を更新する
state0_view = states.view_state_vector_at(0)
state0_view.set_computational_basis(3)
print("batch 0 in states:", states.get_state_vector_at(0).get_amplitudes())

# Copy: detachedでbatch 1を更新する
state1_copy = states.get_state_vector_at(1)
state1_copy.set_computational_basis(2)
print("batch 1 in states:", states.get_state_vector_at(1).get_amplitudes())
print("detached copy:", state1_copy.get_amplitudes())
```
```
batch 0 in states: [0j, 0j, 0j, (1+0j)]
batch 1 in states: [(1+0j), 0j, 0j, 0j]
detached copy: [0j, 0j, (1+0j), 0j]
```

## StateVectorへの操作

{class}`StateVector <scaluq.default.f64.StateVector>` には操作が可能です。

{func}`add_state_vector_with_coef <scaluq.default.f64.add_state_vector_with_coef>` は、 $c\ket{\psi}$ を加えることで状態ベクトルを更新します。ここで、 `c` は複素数、 $\ket{\psi}$ は同じ次元をもつ状態ベクトルです。

{func}`multiply_coef <scaluq.default.f64.multiply_coef>` は、複素数をかけることで状態ベクトルを更新します。

```py
import math
from scaluq.default.f64 import StateVector
phi = StateVector(2)
print("phi:", phi.get_amplitudes())
psi = StateVector.uninitialized_state(2)
psi.set_computational_basis(3)
print("psi:", psi.get_amplitudes())
phi.add_state_vector_with_coef(1j, psi)
print("phi after added psi:", phi.get_amplitudes())
phi.multiply_coef(1 / math.sqrt(2))
print("phi after multiplied coef:", phi.get_amplitudes())
```
```
phi: [(1+0j), 0j, 0j, 0j]
psi: [0j, 0j, 0j, (1+0j)]
phi after added psi: [(1+0j), 0j, 0j, 1j]
phi after multiplied coef: [(0.7071067811865475+0j), 0j, 0j, 0.7071067811865475j]
```

## StateVectorの確率測定

{class}`StateVector <scaluq.default.f64.StateVector>` の確率測度を取得することができます。

{func}`get_zero_probability <scaluq.default.f64.StateVector.get_zero_probability>` は、特定の量子ビットがZ基底で測定された時に $0$ を得る確率を返します。

{func}`get_marginal_probability <scaluq.default.f64.StateVector.get_marginal_probability>` は、いくつかの量子ビットが同時にZ基底で観測された時に特定の結果を得る周辺確率を返します。
結果は長さ `n` のリストの整数値によって指定されます。 $i$ 番目の値の要素は次に従います：
- `0`： $i$ 番目の量子ビットが測定され、結果が $0$である。
- `1`： $i$ 番目の量子ビットが測定され、結果が $1$である。
- {func}`StateVector.UNMEASURED <scaluq.default.f64.StateVector.UNMEASURED>`： $i$ 番目の量子ビットが測定されていない。

{func}`get_entropy <scaluq.default.f64.StateVector.get_entropy>` は、状態ベクトルのエントロピーを返します。 $\sum_i -p_i \log_2 p_i$ ($0\leq i<2^n$) によって計算され、 $p_i$ は$i$番目の状態ベクトルの確率振幅である$v_i$を用いて$|v_i|^2$で与えられます。

{func}`sampling <scaluq.default.f64.StateVector.sampling>` は、状態ベクトルからサンプリングを行います。
サンプリング回数として、`sampling_count` を渡すことで、`sampling_count` 回のサンプリング結果を返します。

```py
import math
from scaluq.default.f64 import StateVector
state = StateVector.uninitialized_state(2)
vec = [1/2, 0, 0, math.sqrt(3)/2 * 1j]
state.load(vec)
print("zero probability of 0:", state.get_zero_probability(0))
assert abs(state.get_zero_probability(0) - (abs(vec[0])**2 + abs(vec[2])**2)) < 1e-9
print("zero probability of 1:", state.get_zero_probability(1))
assert abs(state.get_zero_probability(1) - (abs(vec[0])**2 + abs(vec[1])**2)) < 1e-9
print("marginal probability of [1, UNMEASURED]:", state.get_marginal_probability([1, StateVector.UNMEASURED]))
assert abs(state.get_marginal_probability([1, StateVector.UNMEASURED]) - (abs(vec[1])**2 + abs(vec[3])**2)) < 1e-9
```
```
zero probability of 0: 0.25
zero probability of 1: 0.25
marginal probability of [1, UNMEASURED]: 0.7499999999999999
```
