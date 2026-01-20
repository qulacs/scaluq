# Operator

A quantum operator represents a mathematical object that acts on a quantum state, typically used to represent physical observables (such as energy) or to perform state transformations.

Quantum operators in Scaluq are expressed as {class}Operator <scaluq.default.f64.Operator>, which is composed of one or more {class}PauliOperator <scaluq.default.f64.PauliOperator> instances.

## PauliOperator

{class}PauliOperator <scaluq.default.f64.PauliOperator> represents a tensor product of Pauli operations (I,X,Y,Z) acting on specific qubits, multiplied by a complex coefficient (coef).

Pauli operators can be defined flexibly using strings, lists, or bitmasks.

```py
# 1. Initialization via string
# Applies X to qubit 0 and Y to qubit 2
p1 = PauliOperator("X 0 Y 2", coef=1.0)

# 2. Initialization via target qubit and Pauli ID lists
# IDs: 0=I, 1=X, 2=Y, 3=Z
p2 = PauliOperator(target_qubit_list=[0, 1, 2], pauli_id_list=[1, 0, 2], coef=1.0)

# 3. Initialization via bitmasks
# bit_flip_mask: locations of X, phase_flip_mask: locations of Z
p3 = PauliOperator(bit_flip_mask=0b101, phase_flip_mask=0b010, coef=1.0)
```

## Operator

The {class}Operator <scaluq.default.f64.Operator> class represents a more general operator (such as a Hamiltonian) that is the sum of multiple PauliOperator terms. 

You can use {func}add_operator <scaluq.default.f64.Operator.add_operator> to add terms, or use standard Python arithmetic operators (+, -, *) to construct them intuitively.

```py
from scaluq.default.f64 import Operator, PauliOperator

# Initialize an operator for 2 qubits
n_qubits = 2
op1 = Operator(n_qubits)

# Add a Pauli term
op1.add_operator(PauliOperator("Z 0 Z 1", coef=1.0)) #(1.0 + 0.0j) Z0 Z1

# Use arithmetic operations to add more terms or scale the operator
op1 = op1 + PauliOperator("Z 0 Z 1", coef=0.5) # (1.0 + 0.0j) Z0 Z1 + (0.5 + 0.0j) X0
op1 *= 0.8 # (0.8 + 0.0j) Z0 Z1 + (0.4 + 0.0j) X0


op2 = Operator(n_qubits)
op2.add_operator(PauliOperator("Z 0 Z 1", coef=1.0)) #(1.0 + 0.0j) Z0 Z1
op2.add_operator(PauliOperator("Z 0 Z 1", coef=1.0)) #(1.0 + 0.0j) Z0 Z1 + (1.0 + 0.0j) Z0 Z1
# Optimize to combine identical Pauli terms
op2.optimize()  # (2.0 + 0.0j) Z0 Z1
```

## Calculating Expectation Values and Transition Amplitudes

You can use defined operators to extract physical information from a {class}StateVector <scaluq.default.f64.StateVector>.
Basic Calculations

    Expectation Value: ⟨ψ∣O^∣ψ⟩ Calculated via {func}get_expectation_value <scaluq.default.f64.Operator.get_expectation_value>.

    Transition Amplitude: ⟨ϕ∣O^∣ψ⟩ Calculated via {func}get_transition_amplitude <scaluq.default.f64.Operator.get_transition_amplitude>.


```Python
from scaluq.default.f64 import Operator, PauliOperator, StateVector

n_qubits = 2
op = Operator(n_qubits)
op.add_operator(PauliOperator("Z 0 X 1", coef=1.0))

# Initialize a random state vector using Haar measure
state = StateVector.Haar_random_state(2)

# 1. Get expectation value: <state|op|state>
exp_val = op.get_expectation_value(state)
print(f"Expectation value: {exp_val}") # e.g., (0.16471708718595834+0j)

# 2. Get transition amplitude: <target_state|op|state>
# The first argument is the "Bra" state, and the second is the "Ket" state.
target_state = StateVector.Haar_random_state(2) 
trans_amp = op.get_transition_amplitude(target_state, state) 
print(f"Transition amplitude: {trans_amp}") # e.g., (0.2589420813971237+0.355705558096908j)
```