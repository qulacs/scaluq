from scaluq import *

def main():
    a = PauliOperator([0, 60, 2], [2, 3, 1])
    print(a.get_XZ_mask_representation())

initialize(InitializationSettings().set_num_threads(8))
main()
finalize()
