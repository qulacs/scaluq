import random
import time

def main():
    n_qubits = 20

    import numpy as np
    np.set_printoptions(threshold=np.inf)
    from scaluq.f16 import gate, StateVector
    state = StateVector(2)
    print(state.get_amplitudes())

    # from scaluq.f16 import StateVector, gate
    # state = StateVector(n_qubits)
    # st = time.time()
    # for i in range(10000):
    #     gate.X(random.randint(0, n_qubits-1)).update_quantum_state(state)
    # ed = time.time()
    # print(ed-st)

    # from scaluq.f32 import StateVector, gate
    # state = StateVector(n_qubits)
    # st = time.time()
    # for i in range(10000):
    #     gate.X(random.randint(0, n_qubits-1)).update_quantum_state(state)
    # ed = time.time()
    # print(ed-st)

    # from scaluq.f64 import StateVector, gate
    # state = StateVector(n_qubits)
    # st = time.time()
    # for i in range(10000):
    #     gate.X(random.randint(0, n_qubits-1)).update_quantum_state(state)
    # ed = time.time()
    # print(ed-st)

    # from scaluq.bf16 import StateVector, gate
    # state = StateVector(n_qubits)
    # st = time.time()
    # for i in range(10000):
    #     gate.X(random.randint(0, n_qubits-1)).update_quantum_state(state)
    # ed = time.time()
    # print(ed-st)

if __name__ == "__main__":
    main()
