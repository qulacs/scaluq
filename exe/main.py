from scaluq import *

def main():
    state = StateVector(2)
    gate = SWAP(0, 1)
    gate = SWAPGate(gate)
    mat = gate.get_matrix()
    print(mat)
    t = type(mat[0][0])
    print(t)

if __name__ == "__main__":
    initialize(InitializationSettings().set_num_threads(8))
    main()
    finalize()
