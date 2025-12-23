# Using parametric gates and circuits

## Using parametric gates
Parametric gates allow you to specify arbitrary rotation angles at the time of circuit execution. You can define these gates with an optional `coef` (coefficient) as the second argument.

The actual rotation angle applied is calculated as angle $\times$ coef.

```
from scaluq.default.f64.gate import ParamRX, ParamRY
import math

p_rx = ParamRX(0, 0.5) # Parametric RX gate with target 0 and coef 0.5
p_ry = ParamRY(1) # Parametric RY gate with target 1 and coef 1.0 (default)
```

## Add parametric gates to Circuit
You can add {class}`ParamGate` to `Circuit` using {func}`add_param_gate <scaluq.default.f64.Circuit.add_param_gate>`. This method requires a key (string) to identify the parameter. Multiple gates can share the same key.

You can also retrieve information about parameter keys from the circuit.
```
from scaluq.default.f64.gate import ParamRX, ParamRY, H
from scaluq.default.f64 import Circuit
import math

nqubits = 2
circuit = Circuit(nqubits)

circuit.add_gate(H(0))
circuit.add_param_gate(ParamRX(0), "p_rx")
circuit.add_param_gate(ParamRX(1), "p_rx") # Same key can be used
circuit.add_param_gate(ParamRY(1), "p_ry")

# Get parameter key at specific gate index
print(circuit.get_param_key_at(0)) # None (H gate is not parametric)
print(circuit.get_param_key_at(1)) # p_rx
print(circuit.get_param_key_at(2)) # p_rx
print(circuit.get_param_key_at(3)) # p_ry

# Get the set of all unique parameter keys
print(circuit.key_set()) # {'p_rx', 'p_ry'}
```

## Apply Circuit to StateVector
When executing the circuit, you must provide values for each parameter key. The provided value is applied to all gates with the corresponding key (and multiplied by each gate's coef).

```
from scaluq.default.f64.gate import H, ParamRX, ParamRY
from scaluq.default.f64 import Circuit, StateVector
import math

n_qubits = 2
circuit = Circuit(n_qubits)
state = StateVector(n_qubits) # Initial state |00>

circuit.add_param_gate(ParamRX(0,0.5), "angle_x") # coef 0.5
circuit.add_param_gate(ParamRY(1), "angle_y")
params_1 = {
    "angle_x": math.pi,
    "angle_y": math.pi/2
}

#method 1: using a parameter dictionary
circuit.update_quantum_state(state, params_1)
print(state.get_amplitudes())

#method 2: using keyword arguments
state2 = StateVector(n_qubits)
circuit.update_quantum_state(state2, angle_x = math.pi, angle_y = math.pi/2)
print(state2.get_amplitudes())
```
```
[(0.5000000000000001+0j), -0.5j, (0.5+0j), -0.4999999999999999j]
[(0.5000000000000001+0j), -0.5j, (0.5+0j), -0.4999999999999999j]
```
