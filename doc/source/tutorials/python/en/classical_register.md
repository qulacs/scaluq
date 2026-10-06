# Classical Register

Classical Register is expressed as {class}`scaluq.ClassicalRegister`.

## Create Classical Register
Classical Register is created with register size.

```py
from scaluq import ClassicalRegister

register_size = 2
classical_register = ClassicalRegister(register_size)
print(classical_register.register_size()) # 2
```

## Get classical bit from Quantum Circuit
You can get classical bit by applying Measurement Gate.

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

## Reset Classical Register 
You can reset Classical Register by {func}`reset <scaluq.ClassicalRegister.reset>`.

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
