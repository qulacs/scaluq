# 古典レジスタ

古典レジスタは {class}`scaluq.ClassicalRegister` で表されています。

## 古典レジスタを作る
古典レジスタはレジスタサイズを基に構築されます。

```py
from scaluq import ClassicalRegister

register_size = 2
classical_register = ClassicalRegister(register_size)
print(classical_register.register_size()) # 2
```

## 古典ビットを量子回路から取得する
測定ゲートを作用させることで古典ビットを取得できます。

```py
from scaluq import StateVector, Circuit, ClassicalRegister
from scaluq.gate import X, Measurement

nqubits = 2
circuit = Circuit()
circuit.add_gate(X(0))
circuit.add_gate(Measurement(0, 1))
classical_register = ClassicalRegister(nqubits)

state = StateVector(nqubits)
circuit.update_quantum_state(state, classical_register)
print("classical_register[1]: ", classical_register[1]) # True
```

## 古典レジスタをリセットする
古典レジスタは {func}`reset <scaluq.ClassicalRegister.reset>` によってリセットできます。

```py
from scaluq import StateVector, Circuit, ClassicalRegister
from scaluq.gate import X, Measurement

nqubits = 2
classical_register = ClassicalRegister(nqubits)
state = StateVector(nqubits)

circuit = Circuit()
circuit.add_gate(X(0))
circuit.add_gate(X(1))
circuit.add_gate(Measurement(0, 0))
circuit.add_gate(Measurement(1, 1))

circuit.update_quantum_state(state, classical_register)

print(f"classical_register[0]: {classical_register[0]}, classical_register[1]: {classical_register[1]}") # True
classical_register.reset()
print(f"classical_register[0]: {classical_register[0]}, classical_register[1]: {classical_register[1]}") # False
```
