# 量子回路

量子回路は {class}`Circuit <scaluq.default.f64.Circuit>` で表されています。
量子回路は作用させる {class}`Gate <scaluq.default.f64.Gate>` または {class}`ParamGate <scaluq.default.f64.ParamGate>` のインスタンス配列をもちます。

```{note}
このセクションでは`Circuit`を`Gate`インスタンスの配列とみなします。`ParamGate`の詳しい説明は、[パラメトリックゲートと量子回路を使う](./param.md)を参照してください。
```

Qulacsとは異なり、ゲートを回路の途中に入れること、回路からゲートを取り除くことはできません。これらの操作はユーザーへ複雑なインデックス管理を要請するためです。

## 回路を構築する
回路は引数なしで構築されます。

```py
from scaluq.default.f64 import Circuit

nqubits = 2
circuit = Circuit()
print(circuit.to_json())
```
```
{"gate_list":[]}
```

## 回路にゲートを追加する
{func}`add_gate <scaluq.default.f64.Circuit.add_gate>` により、`Circuit`に`Gate`を追加できます。`Gate`は浅いコピーで追加されます。Scaluqのすべての`Gate`はイミュータブルであり、これは常に安全です。

```py
from scaluq.default.f64 import Circuit
from scaluq.default.f64.gate import H, X

nqubits = 2
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1, controls=[0]))
```

## StateVectorに回路を作用させる
{class}`StateVector <scaluq.default.f64.StateVector>`に{func}`update_quantum_state <scaluq.default.f64.Circuit.update_quantum_state>`を適用することにより、回路を作用させることができます。

```py
from scaluq.default.f64 import Circuit, StateVector
from scaluq.default.f64.gate import H, X

nqubits = 2
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1, controls=[0]))

state = StateVector(nqubits)
circuit.update_quantum_state(state)
print(state.get_amplitudes())
```
```
[(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
```

## Get properties of Circuit
You can get some properties of `Circuit`.
```py
from scaluq.default.f64 import Circuit
from scaluq.default.f64.gate import H, X

nqubits = 2
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(X(1, controls=[0]))

print(circuit.gate_list()) # [H, X, CX]
print(circuit.n_gates()) # 3
print(circuit.calculate_depth()) # 2
```
