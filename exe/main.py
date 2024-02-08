from qulacs2023 import *

def main():
    a = StateVector.Haar_random_state(3)
    print(a)
    i = I(0)
    i.update_quantum_state(a)
    print(a)

initialize(InitializationSettings().set_num_threads(8))
main()
finalize()
