# 量子ゲート

量子ゲートは{class}`scaluq.Gate`と表します。
量子ゲートはターゲットとコントロール量子ビットのインデックス、ゲートの型 (`X`, `H`, `RY`, `DenseMatrix`...)、その他プロパティをもちます。
ゲートのプロパティはその型により異なります。これらのプロパティにアクセスするには、特定クラスへのダウンキャストが必要です。
詳しくは[特定のGateType (ゲート型)へのダウンキャスト](#特定のgatetype-ゲート型へのダウンキャスト)を参照してください。

Qulacsとは異なり、{class}`scaluq.Gate`オブジェクトはイミュータブルです。ゲート作成時にはすべてのプロパティを渡す必要があります。この変更は速さと安全なコピーを可能にしています。

## ゲートを作る

{class}`scaluq.Gate`タイプのインスタンスは{mod}`scaluq.gate`内のファクトリ関数により作られます。

```py
from scaluq.gate import X, Swap, RX, DenseMatrix
import math
import numpy as np

x = X(0) # Xゲート ターゲット量子ビットは0
swap = Swap(2, 4) # Swapゲート ターゲット量子ビットは2, 4
rx = RX(1, math.pi/4) # RX(pi/4)ゲート ターゲット量子ビットは1

mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# 密度行列ゲート 定義したユニタリ行列、ターゲット量子ビットは1, 2
mat_gate = DenseMatrix([1, 2], mat, is_unitary=True)
```

それぞれの型のゲートの作り方の詳しい説明は{mod}`API document of gate module <scaluq.gate>`を参照してください。

ほとんど全てのゲート型で任意の数()のコントロール量子ビット指定できます。
コントロール値も指定することが可能です。省略された場合、全てのコントロール値は $1$ にセットされます。

```py
from scaluq.gate import H

ch = H(0, controls=[1]) # 1番目の量子ビットが|1>の時だけ、H(0)を作用
cch = H(0, controls=[1, 2]) # 1番目の量子ビットが|1>かつ2番目の量子ビットが|1>の時だけ、H(0)を作用
cch = H(0, controls=[1, 2], control_values=[1, 0]) # 1番目の量子ビットが|1>かつ2番目の量子ビットが|0>の時だけ、H(0)を作用
```

## ゲートのプロパティを取得する

一般的なプロパティは{class}`scaluq.Gate`クラスのメソッドを利用して簡単に取得できます。

```py
from scaluq.gate import H

cch = H(0, controls=[1, 2], control_values=[1, 0])
print(cch.target_qubit_list()) # [0]
print(cch.control_qubit_list()) # [1, 2]
print(cch.control_value_list()) # [1, 0]
```

行列は{func}`get_matrix <scaluq.Gate.get_matrix>`で取得できます。Qulacsとは異なり、コントロール量子ビットは無視されます。

```py
from scaluq.gate import H

cch = H(0, controls=[1, 2], control_values=[1, 0])
print(cch.get_matrix())
'''
[[ 0.70710678+0.j  0.70710678+0.j]
 [ 0.70710678+0.j -0.70710678+0.j]]
'''
```

逆行列は{func}`get_inverse <scaluq.Gate.get_inverse>`により取得できます。
逆行列が元のゲートの行列と同一な場合はゲートの浅いコピーが返します。

```py
from scaluq.gate import H, S

h = H(0)
h_inv = h.get_inverse()
print(h_inv)
'''
Gate Type: H
  Target Qubits: {0}
  Control Qubits: {}
  Control Value: {}
'''
s = S(1)
s_inv = s.get_inverse()
print(s_inv)
'''
Gate Type: Sdag
  Target Qubits: {0}
  Control Qubits: {}
  Control Value: {}
'''
```

(特定のgatetype-ゲート型へのダウンキャスト)=
## 特定のGateType (ゲート型)へのダウンキャスト

ゲートごとの詳細なプロパティを取得するには、特定クラスへのダウンキャストが必要になります。

Scaluqのゲートはタグベースのポリモーフィズムで実装されています。それぞれのゲートは{class}`scaluq.GateType`の列挙型として詳細の型をもちます。{func}`gate_type <scaluq.Gate.gate_type>`によってゲート型を取得できます。

型ごとに専用のゲートクラスが定義されており、ダウンキャストを行うことでそれらクラスの固有の関数を利用できます。

```py
from scaluq import GateType, RXGate
from scaluq.gate import RX
import math

rx = RX(0, math.pi/4)
assert rx.gate_type() == GateType.RX
rx = RXGate(rx) # RXGateクラスへの変換
print(rx.angle())
```

これらのクラス間の継承関係はPythonの言語層では見ることができないため、{class}`scaluq.Gate`の型としてゲートを渡すときは、明示的なアップキャストが必要となります。

```py
from scaluq import Gate, RXGate, Circuit
from scaluq.gate import RX
import math

rx = RXGate(RX(0, math.pi/4))
circuit = Circuit()
rx = Gate(rx) # このアップキャストを省略すると次の行でエラーが発生
circuit.add_gate(rx)
```

## StateVectorに作用させる

{func}`update_quantum_state <scaluq.Gate.update_quantum_state>`関数を用いることで{class}`scaluq.StateVector`のインスタンスにゲートを作用させることができます。
ターゲット、コントロール量子ビットのインデックスは状態ベクトルのインデックスと対応しています。

```py
from scaluq import StateVector
from scaluq.gate import H, CX

h0 = H(0)
cx01 = CX(0, 1)
state = StateVector(2)
print(state.get_amplitudes()) # [(1+0j), 0j, 0j, 0j]
h0.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), (0.7071067811865476+0j), 0j, 0j]
cx01.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
```

## 2つのゲートを統合する

2つのゲートを{func}`merge_gate <scaluq.merge_gate>`によって統合可能です。
統合後のゲートの型は以下のように様々です。

```py
from scaluq.gate import X, Y, RX, RY, RZ, S
from scaluq import merge_gate
import math

def print_merge_result(gate1, gate2):
    mgate, phase = merge_gate(gate1, gate2)
    print("Gate:")
    print(mgate.to_string())
    print("Phase:", phase)

print_merge_result(X(0), X(0)) # Gate=I(), Phase=0
print_merge_result(X(0), Y(0)) # Gate=Z(0), Phase=-pi/2
print_merge_result(RZ(0, -math.pi/8*3), S(0)) # Gate=U1(0, math.pi/8), Phase=math.pi/16*3)
print_merge_result(RX(2, math.pi/6, controls=[1]), RY(2, math.pi/3, controls=[0, 1])) # gate=DenseMatrix([0,2], [...], controls=[1], Phase=0)
```
