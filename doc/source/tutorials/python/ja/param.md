# パラメトリックゲートと量子回路を使う

## パラメトリックゲートを使う
パラメトリックゲートは回路作用時に回転角を任意に指定することが可能です。これらのゲートは2番目の引数に`coef` (係数)を与えることで定義できます。

実際に作用させる回転角はangle $\times$ coefで計算されます。

```py
from scaluq.gate import ParamRX, ParamRY
import math

p_rx = ParamRX(0, 0.5) # ターゲットが0、coefが0.5のパラメトリックRXゲート
p_ry = ParamRY(1) # ターゲットが1、coefがデフォルトの1.0となっているパラメトリックRYゲート
```

## 回路にパラメトリックゲートを追加する
{class}`scaluq.Circuit`に{func}`add_param_gate <scaluq.Circuit.add_param_gate>`を使って、{class}`scaluq.ParamGate`を追加することができます。このメソッドはパラメータを指定するために文字列のkeyが必要です。複数のゲートが同じkeyを共有できます。

また、回路からパラメータキーの情報を取得できます。
```py
from scaluq.gate import ParamRX, ParamRY, H
from scaluq import Circuit
import math

nqubits = 2
circuit = Circuit()

circuit.add_gate(H(0))
circuit.add_param_gate(ParamRX(0), "p_rx")
circuit.add_param_gate(ParamRX(1), "p_rx") # 同じキーが使われます
circuit.add_param_gate(ParamRY(1), "p_ry")

# 特定のゲートインデックスからパラメトリックキーを取得する
print(circuit.get_param_key_at(0)) # None (Hゲートはパラメトリックでない)
print(circuit.get_param_key_at(1)) # p_rx
print(circuit.get_param_key_at(2)) # p_rx
print(circuit.get_param_key_at(3)) # p_ry

# 重複なしで全てのパラメトリックキーを取得する
print(circuit.key_set()) # {'p_rx', 'p_ry'}
```

## StateVectorに回路を作用させる

回路を作用させる時、それぞれのパラメータキーに値を与える必要があります。与えられた値は対応するキーをもつすべてのゲートに適用されます。

```py
from scaluq.gate import H, ParamRX, ParamRY
from scaluq import Circuit, StateVector
import math

n_qubits = 2
circuit = Circuit()
state = StateVector(n_qubits) # Initial state |00>

circuit.add_param_gate(ParamRX(0,0.5), "angle_x") # coef 0.5
circuit.add_param_gate(ParamRY(1), "angle_y")
params_1 = {
    "angle_x": math.pi,
    "angle_y": math.pi/2
}

#method 1: using a parameter dictionary
circuit.update_quantum_state(state, params_1)
print(state.get_amplitudes())

#method 2: using keyword arguments
state2 = StateVector(n_qubits)
circuit.update_quantum_state(state2, angle_x = math.pi, angle_y = math.pi/2)
print(state2.get_amplitudes())
```
```
[(0.5000000000000001+0j), -0.5j, (0.5+0j), -0.4999999999999999j]
[(0.5000000000000001+0j), -0.5j, (0.5+0j), -0.4999999999999999j]
```
