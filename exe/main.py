from scaluq.default.f64 import CircuitBuilder
from scaluq.default.f64.gate import H, X
from scaluq.host_serial.f64 import StateVector

nqubits = 2
builder = CircuitBuilder()
builder.add_gate(H(0))
builder.add_gate(X(1))
builder.add_gate(X(1, controls=[0]))
circuit = builder.build()

print(circuit.instructions()[0]) # H
print(circuit.instructions()[1]) # X
print(circuit.instructions()[2]) # CX
print(circuit.n_instructions()) # 3
print(circuit.calculate_depth()) # 2

state = StateVector(nqubits)
circuit.update_quantum_state(state, {})
print(state) # (|00> + |11>)/sqrt(2)

try:
    from scaluq.host_serial.f64 import StateVector
    print("Successfully imported StateVector from scaluq.host_serial.f64")
    v = StateVector(2)
    print(f"StateVector created: {v}")
except ImportError as e:
    print(f"ImportError: {e}")
    exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    exit(1)
