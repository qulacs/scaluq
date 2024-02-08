from qulacs2023 import *

def main():
    a = StateVector(5)
    print(a)

initialize(InitializationSettings().set_num_threads(8))
main()
finalize()
