import random
from scaluq.default.f64 import *

paulis1 = [PauliOperator("X 0 Y 2"), PauliOperator("Z 1 X 3", 2j)]
op1 = Operator(paulis1)
paulis2 = [PauliOperator("X 1", -1j)]
op2 = Operator(paulis2)
op = op1 * op2
op_batched = OperatorBatched([op1, op2])
print(op_batched)
op_batched = op_batched * [-1j, 2.0]
print(op_batched)
