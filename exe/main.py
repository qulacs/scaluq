import random
from qulacs import QuantumState, QuantumCircuit, NoiseSimulator
from qulacs.gate import sqrtX, sqrtY, T, CNOT, X, Z

n = 3
depth = 10
one_qubit_noise = ["BitFlip", "Dephasing"]
circuit = QuantumCircuit(n)


circuit.add_noise_gate(Z(0), "BitFlip", 0.1)

state = QuantumState(n)
state.set_zero_state()
for i in range(1, 1 << n):
    tmp = QuantumState(n);
    tmp.set_computational_basis(i)
    tmp.multiply_coef(i + 1)
    state.add_state(tmp)

sim = NoiseSimulator(circuit, state)

res = sim.execute_and_get_result(1000)

for i in range(res.get_count()):
    print(res.get_frequency(i))
    print(res.get_state(i))
