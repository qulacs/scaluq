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
