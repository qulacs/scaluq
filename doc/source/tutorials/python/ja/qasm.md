# QASM

`scaluq.qasm2` は OpenQASM 2.0 のプログラムをインポートあるいはエクスポートするためのものです。

## OpenQASM 2.0 から量子回路を作成する

OpenQASM 2.0のプログラムから {class}`scaluq.Circuit` をロードできます。

```py
from scaluq import qasm2

result = qasm2.loads('OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; h q[0];')
print(result.n_qubits) # 1
print(result.circuit.n_gates()) # 1
print(result.n_clbits) # 0
print(result.circuit.to_json()) # {"gate_list":[{"gate":{"control":[],"control_value":[],"target":[0],"type":"H"}}]}
```

## 量子回路を OpenQASM 2.0 へ変換する

OpenQASM 2.0 のプログラムを {class}`scaluq.Circuit` に変換することができます。

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
