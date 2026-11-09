# find out best execution space every qubits for best performance in your environment
import os
import pytest
import random
import math
import scaluq as scaluqbase
from scaluq import StateVector, Circuit
import scaluq.gate as mgate

profile_space = os.environ.get("SCALUQ_SPACE")
precision = os.environ.get("SCALUQ_PRECISION")
omp_threads = os.environ.get("OMP_NUM_THREADS")
min_nqubits = int(os.environ.get("MIN_NQUBITS"))
max_nqubits = int(os.environ.get("MAX_NQUBITS"))

default_gpu_backends = {'default(cuda)', 'default(hip)', 'default(sycl)'}
default_cpu_backend = {'default(host)'}
nqubits_list = list(range(min_nqubits, max_nqubits+1))


niter = 100 if profile_space in default_gpu_backends else 10
space = (
    'default'
    if profile_space in (default_gpu_backends | default_cpu_backend)
    else profile_space
)


def benchfunc(circuit, state):
    for _ in range(niter):
        circuit.update_quantum_state(state, {}, 0)
    scaluqbase.synchronize()

@pytest.mark.parametrize("nqubits", nqubits_list)
def test(benchmark, nqubits):
    random.seed(nqubits)
    benchmark.group = 'Single-StateVector-Circuit'
    benchmark.extra_info["niter"] = niter
    benchmark.extra_info["profile_space"] = profile_space
    benchmark.extra_info["precision"] = precision
    benchmark.extra_info["omp_threads"] = omp_threads
    circuit = Circuit(precision=precision)
    for i in range(nqubits):
        circuit.add_gate(mgate.CX(i, (i+1) % nqubits, precision=precision))
        circuit.add_gate(mgate.RX(i, random.uniform(0, math.pi * 2), precision=precision))
        circuit.add_gate(mgate.RZ(i, random.uniform(0, math.pi * 2), precision=precision))
    state = StateVector(nqubits, precision=precision, space=space)
    benchfunc(circuit, state)  # warmup
    benchmark(benchfunc, circuit, state)

