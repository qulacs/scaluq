from scaluq.default.f64 import StateVector
from scaluq.default.f64.gate import H, CX

h0 = H(0)
cx01 = CX(0, 1)
state = StateVector(2)
print(state.get_amplitudes()) # [(1+0j), 0j, 0j, 0j]
h0.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), (0.7071067811865476+0j), 0j, 0j]
cx01.update_quantum_state(state)
print(state.get_amplitudes()) # [(0.7071067811865476+0j), 0j, 0j, (0.7071067811865476+0j)]
