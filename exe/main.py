from scaluq.default.f64 import Circuit
from scaluq.default.f64.gate import H, X
from scaluq.host_serial.f64 import StateVector

nqubits = 2
circuit = Circuit(nqubits)
circuit.add_gate(H(0))
circuit.add_gate(X(1))
circuit.add_gate(X(1, controls=[0]))

print(circuit.n_qubits()) # 2
print(circuit.gate_list()[0]) # [H, X, CX]
print(circuit.gate_list()[1]) # [H, X, CX]
print(circuit.gate_list()[2]) # [H, X, CX]
print(circuit.n_gates()) # 3
print(circuit.calculate_depth()) # 2

state = StateVector(nqubits)
circuit.update_quantum_state(state, {})
print(state) # (|00> + |11>)/sqrt(2)
