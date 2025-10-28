# Gate

Quantum gate is expressed as {class}`Gate <scaluq.default.f64.Gate>`.
This holds indices of target and control qubits, types of Gate (ex: `X`, `H`, `RY`, `DenseMatrix`...), and other properties.
Properties of a Gate differ by its type. To access these properties, down-casting to specific class is required. (See [Downcast to GateType-specific function](#downcast-to-gatetype-specific-function))

Unlike Qulacs, {class}`Gate <scaluq.default.f64.Gate>` objects are immutable. You have to pass all properties when generating the gate. This change enables us to provide fast and safe copy.

## Creating Gate

{class}`Gate <scaluq.default.f64.Gate>` type instances are created from factory functions in {mod}`gate <scaluq.default.f64.gate>`.

```py
from scaluq.default.f64.gate import X, Swap, RX, DenseMatrix
import math
import numpy as np

x = X(0) # X gate with target 0
swap = Swap(2, 4) # Swap gate with target 2, 4
rx = RX(1, math.pi/4) # RX(pi/4) gate with target 1

mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# DenseMatrix gate with certain unitary matrix and target 1,2
mat_gate = DenseMatrix([1, 2], mat, is_unitary=True)
```

For detail usage of creating each types of gate, see {mod}`API document of gate module <scaluq.default.f64.gate>`.

You can add arbitrary number of control qubits to almost all types of Gates.
Control-values can be passed. All control-values are set to $1$ when omitted.

```py
from scaluq.default.f64.gate import H

ch = H(0, controls=[1]) # H(0) applied when 1st qubit is |1>
cch = H(0, controls=[1, 2]) # H(0) applied when 1st qubit is |1> and 2nd qubit is |1>
cch = H(0, controls=[1, 2], control_values=[1, 0]) # H(0) applied when 1st qubit is |1> and 2nd qubit is |0>
```

## Getting properties of Gate.
General properties are simply gotten by using methods of {class}`Gate <scaluq.default.f64.Gate>` class.

```py
from scaluq.default.f64.gate import H

cch = H(0, controls=[1, 2], control_values=[1, 0])
print(cch.target_qubit_list()) # [0]
print(cch.control_qubit_list()) # [1, 2]
print(cch.control_value_list()) # [1, 0]
```

Matrix representation is showed by {func}`get_matrix <scaluq.default.f64.Gate.get_matrix>`. Unlike Qulacs, control qubits are ignored.

```py
from scaluq.default.f64.gate import H

cch = H(0, controls=[1, 2], control_values=[1, 0])
print(cch.get_matrix())
'''
[[ 0.70710678+0.j  0.70710678+0.j]
 [ 0.70710678+0.j -0.70710678+0.j]]
'''
```

Inverse gate is gotten by {func}`get_inverse <scaluq.default.f64.Gate.get_inverse>`.
If inverse gate is completely same as the original gate, the gate is shallow-copied.
```py
from scaluq.default.f64.gate import H, S

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

## Downcast to GateType-specific function
To get GateType-specific properties, downcast to specific class is required.

Scaluq Gate is implemented by tag-based polymorphism. Each Gate has detailed type as enum {class}`GateType <scaluq.GateType>`. You can get type of Gate by {func}`get_type <scaluq.default.f64.Gate.gate_type>`.

There is a specific Gate class for each type. You can use functions of these specific types by downcast.

```py
from scaluq import GateType
from scaluq.default.f64 import RXGate
from scaluq.default.f64.gate import RX
import math

rx = RX(0, math.pi/4)
assert rx.gate_type() == GateType.RX
rx = RXGate(rx) # downcast to RXGate class
print(rx.angle())
```

Since this inheritance relation is not shown in language layer, explicit upcast is required when you pass the Gate as {class}`Gate <scaluq.default.f64.Gate>` type.

```py
from scaluq.default.f64 import Gate, RXGate, Circuit
from scaluq.default.f64.gate import RX
import math

rx = RXGate(RX(0, math.pi/4))
circuit = Circuit(1)
rx = Gate(rx) # omitting this downcast causes error on next line
circuit.add_gate(rx)
```

## Apply to StateVector
Gate can be applied to {class}`StateVector <scaluq.default.f64.StateVector>` object by function {func}`update_quantum_state <scaluq.default.f64.Gate.update_quantum_state>`.
Indices of Target/control qubits of Gate are corresponded to that of StateVector.

```py
from scaluq.default.f64 import StateVector
from scaluq.default.f64.gate import H, CX

h0 = H(0)
cx01 = CX(0, 1)
state = StateVector(2)
print(state.get_amplitudes()) # [(1+0j), 0j, 0j, 0j]
h0.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), (0.7071067811865476+0j), 0j, 0j]
cx01.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
```

## Merge two Gates
You can merge two Gates by {func}`merge_gate <scaluq.default.f64.gate.merge_gate>`
The type of result gate is flexible.

```py
from scaluq.default.f64.gate import X, Y, RX, RY, RZ, S
from scaluq.default.f64 import merge_gate
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
