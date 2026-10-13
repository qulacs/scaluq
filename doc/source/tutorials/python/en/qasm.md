# QASM

`scaluq.qasm2` is for importing and exporting OpenQASM 2.0 programs.

## Create Circuit from OpenQASM 2.0

You can loads {class}`scaluq.Circuit` from OPENQASM 2.0 program.

```py
from scaluq import qasm2

result = qasm2.loads('OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];')
print(result.n_qubits) # 1
print(result.circuit.n_gates()) # 1
print(result.n_clbits) # 0
print(result.circuit.to_json()) # {"gate_list":[{"gate":{"control":[],"control_value":[],"target":[0],"type":"H"}}]}
```

## Circuit to OpenQASM 2.0

You can turn {class}`scaluq.Circuit` into OPENQASM 2.0 program.

```py
from scaluq import Circuit
from scaluq import gate, qasm2

circuit = Circuit()
circuit.add_gate(gate.H(0))
print(qasm2.dumps(circuit, 1))
```
```
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
```
