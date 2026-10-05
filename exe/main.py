from scaluq import StateVector, Circuit
from scaluq import gate

nqubits = 2
circuit = Circuit()
circuit.add_gate(gate.H(0))
circuit.add_gate(gate.X(1))
circuit.add_gate(gate.X(1, controls=[0]))

print(circuit.gate_list()[0]) # H...
print(circuit.gate_list()[1]) # X...
print(circuit.gate_list()[2]) # X...
print(circuit.n_gates()) # 3
print(circuit.calculate_depth()) # 2

# StateVector of (precision, space) = (f64, default)
state = StateVector(nqubits)
circuit.update_quantum_state(state, {})
print(state) # (|00> + |11>)/sqrt(2)

# StateVector of (precision, space) = (f64, host_serial)
v = StateVector(2, precision='f64', space='host_serial')
print(f"StateVector created: {v}")
