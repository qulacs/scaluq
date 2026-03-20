# Batched Execution 
In many quantum algorithms, such as VQE or quantum machine learning, processing multiple quantum states or operators simultaneously can significantly improve performance. Scaluq provides the {class}`StateVectorBatched <scaluq.default.f64.StateVectorBatched>` and {class}`OperatorBatched <scaluq.default.f64.OperatorBatched>` classes to handle these operations efficiently in a single batch.

## Create Batched State

To begin with, you can initialize a batched state vector by specifying the number of `batch_size` and `n_qubits`.

```
from scaluq.default.f64 import StateVectorBatched

batch_size = 3
n_qubits = 2

# Initialize 3 state vectors, each with 2 qubits
states = StateVectorBatched(batch_size, n_qubits)
print(states)
```
```
Qubit Count : 1
Dimension : 2
--------------------
Batch_id : 0
State vector : 
  0 : (1,0)
  1 : (0,0)
--------------------
Batch_id : 1
State vector : 
  0 : (0.707107,0)
  1 : (0,-0.707107)
```


## Apply Circuit to Batched States

You can apply a {class}`Circuit <scaluq.default.f64.Circuit>` to a {class}`StateVectorBatched <scaluq.default.f64.StateVectorBatched>`. This allows you to apply the same gate operations to multiple states simultaneously.

### Set Different States

You can prepare different initial states for each batch index using {func}`set_state_vector_at <scaluq.default.f64.StateVectorBatched>`.

```
from scaluq.default.f64 import Circuit, StateVectorBatched, StateVector
from scaluq.default.f64.gate import H

n_qubits = 2
states = StateVectorBatched(batch_size=2, n_qubits=n_qubits)

# Set different random initial states for each batch
states.set_state_vector_at(0, StateVector.Haar_random_state(n_qubits)) # Batch 0: Random state
states.set_state_vector_at(1, StateVector.Haar_random_state(n_qubits)) # Batch 1: Another random state

# Define a non-parametric circuit
circuit = Circuit(n_qubits)
circuit.add_gate(H(0))

# Apply the same circuit to both different random states
circuit.update_quantum_state(states)
print(states)
```

```
Qubit Count : 2
Dimension : 4
--------------------
Batch_id : 0
State vector : 
  00 : (0.321818,0.191616)
  01 : (-0.114258,-0.549058)
  10 : (-0.554059,0.443414)
  11 : (-0.139882,-0.148433)
--------------------
Batch_id : 1
State vector : 
  00 : (-0.156496,0.393741)
  01 : (0.111451,0.716401)
  10 : (-0.0756985,-0.190015)
  11 : (0.370353,0.340334)
```

### Parametric Execution
When the circuit is parametric, you can execute circuits with different parameter values across the batch.

```
from scaluq.default.f64 import Circuit, StateVectorBatched
from scaluq.default.f64.gate import ParamRX
import math

states = StateVectorBatched(batch_size=2, n_qubits=1)

circuit = Circuit(1)
# set a parameterized RX gate on qubit 0 with parameter name "theta"
circuit.add_param_gate(ParamRX(0), "theta")

# Batch 0: theta=0.0, Batch 1: theta=pi/2 
circuit.update_quantum_state(states, theta=[0.0, math.pi / 2])

print(states)
```

```
Qubit Count : 1
Dimension : 2
--------------------
Batch_id : 0
State vector : 
  0 : (1,0)
  1 : (0,0)
--------------------
Batch_id : 1
State vector : 
  0 : (0.707107,0)
  1 : (0,-0.707107)

```

## Batched Operators
(Content to be added)
