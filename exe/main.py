import matplotlib.pyplot as plt
import numpy as np

# Data for batchsize=1000
n_qubits = np.arange(1, 19)
layout_left_cpu = [
    5.11337e-05, 5.40003e-05, 5.9467e-05, 0.000113051, 0.000122851, 0.000177118, 
    0.000291652, 0.00110281, 0.00250263, 0.00545675, 0.0140253, 0.0341415, 
    0.0849506, 0.221406, 0.533939, 1.11097, 2.34909, 4.8307
]
layout_right_cpu = [
    4.35502e-05, 5.0717e-05, 5.46503e-05, 8.49673e-05, 8.8434e-05, 0.000129834, 
    0.00048782, 0.000540887, 0.000774889, 0.00254077, 0.0068662, 0.0164833, 
    0.0357408, 0.0620327, 0.121985, 0.242586, 0.480349, 0.974581
]
serial_cpu = [
    1.46766, 1.89952, 1.89428, 1.87931, 1.92239, 2.1463, 
    1.88771, 1.93818, 2.07707, 2.12035, 2.0031, 1.74588, 
    1.79551, 1.82069, 1.84837, 2.27512, 1.83291, 2.09968
]
serial_all_mem_cpu = [
    2.34921, 5.66175, 0.0301746, 0.0308457, 0.0366038, 0.0294894, 
    0.0374785, 0.0334359, 0.0338106, 0.0370775, 0.0480555, 0.0575162, 
    0.101832, 0.179661, 0.330166, 0.634618, 1.24223, 3.66223
]

layout_left_gpu = [
    0.00325633, 0.00411396, 0.00343683, 0.0037027, 0.00428346, 0.00364633, 
    0.00344442, 0.00525013, 0.00312933, 0.0131786, 0.0189822, 0.0259635, 
    0.0460142, 0.060699, 0.0782006, 0.112535, 0.155156, 0.327711
]
layout_right_gpu = [
    0.00301848, 0.00462549, 0.00438684, 0.00326001, 0.00405515, 0.00414581, 
    0.00330985, 0.00490436, 0.00307708, 0.00916117, 0.0166353, 0.0241079, 
    0.013927, 0.0230415, 0.0410164, 0.0842886, 0.11416, 0.220944
]
serial_gpu = [
    1.46766, 1.89952, 1.89428, 1.87931, 1.92239, 2.1463, 
    1.88771, 1.93818, 2.07707, 2.12035, 2.0031, 1.74588, 
    1.79551, 1.82069, 1.84837, 2.27512, 1.83291, 2.09968
]
serial_all_mem_gpu = [
    1.88925, 1.82697, 1.85054, 1.79228, 1.81207, 1.84095, 
    1.82022, 1.98252, 2.01957, 1.94565, 1.80599, 1.84642, 
    1.88757, 1.95968, 2.11865, 1.84444, 1.77062, 2.16731
]

plt.figure(figsize=(12, 8))
plt.plot(n_qubits, layout_left_cpu, label='LayoutLeft CPU')
plt.plot(n_qubits, layout_right_cpu, label='LayoutRight CPU')
plt.plot(n_qubits, serial_cpu, label='Serial CPU')
plt.plot(n_qubits, serial_all_mem_cpu, label='Serial (all memory) CPU')
plt.plot(n_qubits, layout_left_gpu, label='LayoutLeft GPU')
plt.plot(n_qubits, layout_right_gpu, label='LayoutRight GPU')
plt.plot(n_qubits, serial_gpu, label='Serial GPU')
plt.plot(n_qubits, serial_all_mem_gpu, label='Serial (all memory) GPU')

plt.yscale('log')
plt.xlabel('Number of Qubits')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Number of Qubits (Batch size = 1000)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('execution_time_vs_qubits.png')
