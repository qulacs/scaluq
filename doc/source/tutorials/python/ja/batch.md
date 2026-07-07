# バッチ実行
数多くの量子アルゴリズム (VQE, quantum machine learningなど)では複数の量子状態または演算子の処理を同時に進めることでパフォーマンスが大幅に改善します。Scaluqは{class}`scaluq.StateVectorBatched` および {class}`scaluq.OperatorBatched`でバッチ実行を提供し、これらの操作を効率的に扱うことが可能です。

## バッチ状態を作る

バッチサイズと量子ビット数を指定することでバッチ状態ベクトルを初期化できます。

```py
from scaluq import StateVectorBatched

batch_size = 3
n_qubits = 2

# それぞれの状態ベクトルが2量子ビットをもつ、3つのバッチ状態ベクトルに初期化
states = StateVectorBatched(batch_size, n_qubits)
print(states)
```
```
Qubit Count : 2
Dimension : 4
--------------------
Batch_id : 0
State vector : 
  00 : (1,0)
  01 : (0,0)
  10 : (0,0)
  11 : (0,0)
--------------------
Batch_id : 1
State vector : 
  00 : (1,0)
  01 : (0,0)
  10 : (0,0)
  11 : (0,0)
--------------------
Batch_id : 2
State vector : 
  00 : (1,0)
  01 : (0,0)
  10 : (0,0)
  11 : (0,0)
```


## バッチ量子状態に回路を作用させる

{class}`scaluq.StateVectorBatched`に{class}`scaluq.Circuit`を作用させることができます。これにより、同じゲート操作を複数の状態に同時に作用可能にします。

### 異なる量子状態にセットする

{func}`set_state_vector_at <scaluq.StateVectorBatched.set_state_vector_at>`を使って、バッチインデックスごとに異なる初期状態を準備することができます。

```py
from scaluq import Circuit, StateVectorBatched, StateVector
from scaluq.gate import H

n_qubits = 2
states = StateVectorBatched(batch_size=2, n_qubits=n_qubits)

# バッチごとに異なるランダムな初期状態にセット
states.set_state_vector_at(0, StateVector.Haar_random_state(n_qubits)) # Batch 0: Random state
states.set_state_vector_at(1, StateVector.Haar_random_state(n_qubits)) # Batch 1: Another random state

# パラメトリックでない回路を定義
circuit = Circuit()
circuit.add_gate(H(0))

# 異なるランダムな状態に同じ回路を作用
circuit.update_quantum_state(states)
print(states)
```

```
Qubit Count : 2
Dimension : 4
--------------------
Batch_id : 0
State vector : 
  00 : (0.321818,0.191616)
  01 : (-0.114258,-0.549058)
  10 : (-0.554059,0.443414)
  11 : (-0.139882,-0.148433)
--------------------
Batch_id : 1
State vector : 
  00 : (-0.156496,0.393741)
  01 : (0.111451,0.716401)
  10 : (-0.0756985,-0.190015)
  11 : (0.370353,0.340334)
```

### パラメトリック実行

回路がパラメトリックである時、異なるパラメータをもつ回路をバッチ実行可能です。

```py
from scaluq import Circuit, StateVectorBatched
from scaluq.gate import ParamRX
import math

states = StateVectorBatched(batch_size=2, n_qubits=1)

circuit = Circuit()
# パラメトリックRXゲートを量子ビットのインデックス0にthetaというパラメータを用いてセット
circuit.add_param_gate(ParamRX(0), "theta")

# Batch 0: theta=0.0, Batch 1: theta=pi/2 
circuit.update_quantum_state(states, theta=[0.0, math.pi / 2])

print(states)
```

```
Qubit Count : 1
Dimension : 2
--------------------
Batch_id : 0
State vector : 
  0 : (1,0)
  1 : (0,0)
--------------------
Batch_id : 1
State vector : 
  0 : (0.707107,0)
  1 : (0,-0.707107)

```

## 演算子のバッチ

{class}`scaluq.OperatorBatched`では、単一の量子状態に対して、複数の異なる演算子で期待値の計算可能です。

```py
from scaluq import OperatorBatched, PauliOperator, StateVector

# ハール測度を用いてランダムな状態ベクトルに初期化
state = StateVector.Haar_random_state(2)

# 期待値を取得
pauli1 = PauliOperator("Z 0 X 1", coef=1.0)
pauli2 = PauliOperator("X 0 Z 1", coef=0.5)
pauli3 = PauliOperator("Y 0 Y 1", coef=0.8)
op = OperatorBatched([[pauli1], [pauli2], [pauli3]]) # 3つの演算子をまとめる

exp_val = op.get_expectation_value(state)
print(f"Expectation value: {exp_val}") # e.g., [(-0.09541586059802383+0j), (-0.0537398701798849+0j), (0.47697004250289277+0j)]
```
