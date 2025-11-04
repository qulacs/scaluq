from scaluq.default.f64.gate import X, Y, RX, RY, RZ, S
from scaluq.default.f64 import merge_gate
import math

def print_merge_result(gate1, gate2):
    mgate, phase = merge_gate(gate1, gate2)
    print("Gate:")
    print(mgate.to_string())
    print("Phase:", phase)

print_merge_result(X(0), X(0)) # Gate=I(), Phase=0
print_merge_result(X(0), Y(0)) # Gate=Z(0), Phase=-pi/2
print_merge_result(RZ(0, -math.pi/8*3), S(0)) # Gate=U1(0, math.pi/8), Phase=math.pi/16*3)
print_merge_result(RX(2, math.pi/6, controls=[1]), RY(2, math.pi/3, controls=[0, 1])) # gate=DenseMatrix([0,2], [...], controls=[1], Phase=0)
