# Operator

A quantum operator represents a mathematical object that acts on a quantum state, typically used to represent physical observables (such as energy) or to perform state transformations.

Quantum operators in Scaluq are expressed as {class}`scaluq.Operator`, which is composed of one or more {class}`scaluq.PauliOperator` instances.

## PauliOperator

{class}`scaluq.PauliOperator` represents a tensor product of Pauli operations (I,X,Y,Z) acting on specific qubits, multiplied by a complex coefficient (coef).

Pauli operators can be defined flexibly using strings, lists, or bitmasks.

```py
from scaluq import PauliOperator

# 1. Initialization via string
# Applies X to qubit 0 and Y to qubit 2
p1 = PauliOperator("X 0 Y 2", coef=1.0)

# 2. Initialization via target qubit and Pauli ID lists
# IDs: 0=I, 1=X, 2=Y, 3=Z
p2 = PauliOperator.from_targets_and_pauli_ids([0, 1, 2], [1, 0, 2], coef=1.0)

# 3. Initialization via bitmasks
# bit_flip_mask: locations of X, phase_flip_mask: locations of Z
p3 = PauliOperator.from_XZ_mask(0b101, 0b010, coef=1.0)
```

## Operator

The {class}`scaluq.Operator` represents a general operator (such as a Hamiltonian) that is defined as a sum of multiple {class}`scaluq.PauliOperator` terms.

```py
from scaluq import Operator, PauliOperator

#prepare two pauli
pauli1 = PauliOperator("Z 0 Z 1", coef=0.5) # (0.5 + 0.0j) Z0 Z1
pauli2 = PauliOperator("Z 0 Z 1", coef=0.3) # (0.3 + 0.0j) Z0 Z1
terms = [pauli1, pauli2]

op1 = Operator(terms)
print(op1) # (0.5 + 0.0j) Z0 Z1，(0.3 + 0.0j) Z0 Z1

# you can optimize the operator by merging terms with the same Pauli string
op1.optimize()
print(op1) # (0.8 + 0.0j) Z0 Z1
```

## Calculating Expectation Values and Transition Amplitudes

You can use defined operators to extract physical information from a {class}`scaluq.StateVector`.

```py
from scaluq import Operator, PauliOperator, StateVector

# Initialize a random state vector using Haar measure
state = StateVector.Haar_random_state(2)

# 1. Get expectation value: <state|op|state>
pauli = PauliOperator("Z 0 X 1", coef=1.0)
op = Operator([pauli])
exp_val = op.get_expectation_value(state)
print(f"Expectation value: {exp_val}") # e.g., (0.16471708718595834+0j)

# 2. Get transition amplitude: <target_state|op|state>
# The first argument is the "Bra" state, and the second is the "Ket" state.
target_state = StateVector.Haar_random_state(2) 
trans_amp = op.get_transition_amplitude(target_state, state) 
print(f"Transition amplitude: {trans_amp}") # e.g., (0.2589420813971237+0.355705558096908j)
```

### Grouped Pauli sums

`GroupedPauliOperator(n_qubits, terms, cache_budget=0)` takes a snapshot of a
list of `PauliOperator`s and groups terms with the same X/Y bit-flip mask.
`apply_to_state(input, output, coef=1)` **adds** `coef * H * input` to
`output`; input and output may refer to the same state. The class also provides
`get_expectation_value(state)` and `get_transition_amplitude(bra, ket)`, both
of which retain complex coefficients. No symmetry-based terms are discarded.

The cache budget is in bytes and covers the diagonal weight arrays, excluding
term metadata. If all arrays fit, construction takes O(L 2^n) work and each
application or transition takes O(G 2^n), for L terms and G groups. Otherwise,
the class streams the terms, taking O(L 2^n) work per call. The default budget
is zero. `n_groups()` and `is_cached()` expose the selected representation.
