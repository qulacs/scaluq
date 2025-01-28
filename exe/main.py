from scaluq.f64 import *

def main():
    state = StateVector(1)
    x_gate = gate.X(0, [1])
    print(state.to_json())
    print(x_gate.to_json())
    print(gate.ParamRX(0, 0.5, [1]).to_json())

    circuit = Circuit(3)
    circuit.add_gate(x_gate)
    print(circuit.to_json())

    pauli = PauliOperator("X 3 Y 2")

    operator = Operator(4)
    operator.add_operator(pauli)

    print(operator.to_json())

    states = StateVectorBatched.Haar_random_state(2, 3)
    prx = gate.ParamRX(0, 2.0, [1])
    pry = gate.ParamRY(1, 2.0, [2])
    params = {
        "rx": [0.0, 0.1, 0.2],
        "ry": [0.3, 0.4, 0.5]
    }
    circuit.add_param_gate(prx, "rx")
    circuit.add_param_gate(pry, "ry")
    circuit.update_quantum_state(states, params)
    print(states.to_json())

if __name__ == "__main__":
    main()
