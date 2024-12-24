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

    states = StateVectorBatched(3, 3)
    print(states.to_json())


if __name__ == "__main__":
    main()
