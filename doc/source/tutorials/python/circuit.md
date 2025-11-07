# Circuit

Quantum circuit is expressed as {class}`Circuit <scaluq.default.f64.Circuit>`.
This holds an array of instances of {class}`Gate <scaluq.default.f64.Gate>` or {class}`ParamGate <scaluq.default.f64.ParamGate>` to be applied. 

```{note}
In this section, we regard `Circuit` as an array of `Gate` instances. To learn about `ParamGate`, see [Using parametric gates and circuits](./param.md).
```

Unlike Qulacs, you cannot insert Gate in the middle of Circuit or remove Gate from Circuit. These operations make user do complicated index-management.

## Create Circuit
Circuit is created by passing number of qubits as single argument.

```py
from scaluq.default.f64 import Circuit

nqubits = 2
circuit = Circuit(nqubits)
print(circuit.to_json())
```
```
{"gate_list":[],"n_qubits":2}
```

## Add Gate to Circuit
You can add `Gate` to `Circuit` by {func}`add_gate <scaluq.default.f64.Circuit.add_gate>`. `Gate` to be added is shallow-copied. Since all the `Gate`s in Scaluq are immutable, this is always safe!

```py
from scaluq.default.f64 import Circuit
from scaluq.default.f64.gate import H, X

nqubits = 2
circuit = Circuit(nqubits)
circuit.add_gate(H(0))
circuit.add_gate(X(1, controls=[0]))
```

## Apply Circuit to StateVector
You can run Circuit by applying {func}`update_quantum_state <scaluq.default.f64.Circuit.update_quantum_state>` to {class}`StateVector <scaluq.default.f64.StateVector>`.

```py
from scaluq.default.f64 import Circuit, StateVector
from scaluq.default.f64.gate import H, X

nqubits = 2
circuit = Circuit(nqubits)
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
circuit = Circuit(nqubits)
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(X(1, controls=[0]))

print(circuit.n_qubits()) # 2
print(circuit.gate_list()) # [H, X, CX]
print(circuit.n_gates()) # 3
print(circuit.calculate_depth()) # 2
```
