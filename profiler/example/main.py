from scaluq import StateVector
import profiler

nqubits = 14
prec = "f64"

profile = profiler.Profile()
space = profile.get_fast_space(nqubits, prec)

state = StateVector(n_qubits=nqubits, precision=prec, space=space)
