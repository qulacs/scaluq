import math

import scaluq
from scaluq import gate


def test_density_matrix_circuit_accepts_positional_parameters():
    circuit = scaluq.Circuit()
    circuit.add_param_gate(gate.ParamRX(0), "angle")
    dm = scaluq.DensityMatrix(1)

    circuit.update_quantum_state(dm, {"angle": math.pi})

    assert abs(dm.get_coherence_at(1, 1) - 1) < 1e-6


def test_density_matrix_circuit_accepts_positional_classical_register():
    circuit = scaluq.Circuit()
    circuit.add_gate(gate.X(0))
    circuit.add_gate(gate.Measurement(0, 0))
    dm = scaluq.DensityMatrix(1)
    reg = scaluq.ClassicalRegister(1)

    circuit.update_quantum_state(dm, reg)

    assert reg[0] == 1
