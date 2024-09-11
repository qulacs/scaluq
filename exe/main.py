from scaluq import *
# from scaluq.gate import S

def main():
    state = StateVector(2)
    swap_gate = gate.Swap(0, 1)
    print(state)
    print(swap_gate)

if __name__ == "__main__":
    main()
