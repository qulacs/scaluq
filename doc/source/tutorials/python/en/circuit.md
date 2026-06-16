# Circuit

Quantum circuit is expressed as {class}`scaluq.Circuit`.
This holds an array of instances of {class}`scaluq.Gate` or {class}`scaluq.ParamGate` to be applied. 

```{note}
In this section, we regard `Circuit` as an array of `Gate` instances. To learn about `ParamGate`, see [Using parametric gates and circuits](./param.md).
```

Unlike Qulacs, you cannot insert Gate in the middle of Circuit or remove Gate from Circuit. These operations make user do complicated index-management.

## Create Circuit
Circuit is created without arguments.

```py
from scaluq import Circuit

nqubits = 2
circuit = Circuit()
print(circuit.to_json())
```
```
{"gate_list":[]}
```

## Add Gate to Circuit
You can add `Gate` to `Circuit` by {func}`add_gate <scaluq.Circuit.add_gate>`. `Gate` to be added is shallow-copied. Since all the `Gate`s in Scaluq are immutable, this is always safe!

```py
from scaluq import Circuit
from scaluq.gate import H, X

nqubits = 2
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1, controls=[0]))
```

## Apply Circuit to StateVector
You can run Circuit by applying {func}`update_quantum_state <scaluq.Circuit.update_quantum_state>` to {class}`scaluq.StateVector`.

```py
from scaluq import Circuit, StateVector
from scaluq.gate import H, X

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
from scaluq import Circuit
from scaluq.gate import H, X

nqubits = 2
circuit = Circuit()
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(X(1, controls=[0]))

print(circuit.gate_list()) # [H, X, CX]
print(circuit.n_gates()) # 3
print(circuit.calculate_depth()) # 2
```
