:py:mod:`scaluq`
================

.. py:module:: scaluq


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   scaluq.CXGate
   scaluq.CZGate
   scaluq.Circuit
   scaluq.FusedSwapGate
   scaluq.Gate
   scaluq.GateType
   scaluq.GlobalPhaseGate
   scaluq.HGate
   scaluq.IGate
   scaluq.InitializationSettings
   scaluq.OneQubitMatrixGate
   scaluq.Operator
   scaluq.P0Gate
   scaluq.P1Gate
   scaluq.PauliGate
   scaluq.PauliOperator
   scaluq.PauliRotationGate
   scaluq.RXGate
   scaluq.RYGate
   scaluq.RZGate
   scaluq.SGate
   scaluq.SdagGate
   scaluq.SqrtXGate
   scaluq.SqrtXdagGate
   scaluq.SqrtYGate
   scaluq.SqrtYdagGate
   scaluq.StateVector
   scaluq.SwapGate
   scaluq.TGate
   scaluq.TdagGate
   scaluq.TwoQubitMatrixGate
   scaluq.U1Gate
   scaluq.U2Gate
   scaluq.U3Gate
   scaluq.XGate
   scaluq.YGate
   scaluq.ZGate



Functions
~~~~~~~~~

.. autoapisummary::

   scaluq.CNot
   scaluq.CX
   scaluq.CZ
   scaluq.FusedSwap
   scaluq.GlobalPhase
   scaluq.H
   scaluq.I
   scaluq.P0
   scaluq.P1
   scaluq.Pauli
   scaluq.PauliRotation
   scaluq.RX
   scaluq.RY
   scaluq.RZ
   scaluq.S
   scaluq.Sdag
   scaluq.SqrtX
   scaluq.SqrtXdag
   scaluq.SqrtY
   scaluq.SqrtYdag
   scaluq.Swap
   scaluq.T
   scaluq.Tdag
   scaluq.U1
   scaluq.U2
   scaluq.U3
   scaluq.X
   scaluq.Y
   scaluq.Z
   scaluq.finalize
   scaluq.initialize
   scaluq.is_finalized
   scaluq.is_initialized



.. py:function:: CNot(arg0: int, arg1: int, /) -> Gate

   Generate general Gate class instance of CX.
   [note] CNot is an alias of CX.


.. py:function:: CX(arg0: int, arg1: int, /) -> Gate

   Generate general Gate class instance of CX.


.. py:class:: CXGate(arg: Gate, /)


       Specific class of single-qubit-controlled Pauli-X gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: control() -> int

      Get property `control`.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int

      Get property `target`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: CZ(arg0: int, arg1: int, /) -> Gate

   Generate general Gate class instance of CZ.


.. py:class:: CZGate(arg: Gate, /)


       Specific class of single-qubit-controlled Pauli-Z gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: control() -> int

      Get property `control`.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int

      Get property `target`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: Circuit(arg: int, /)


   Quantum circuit represented as gate array

   .. py:method:: add_circuit(arg: Circuit, /) -> None

      Add all gates in specified circuit. Given gates are copied.


   .. py:method:: add_gate(arg: Gate, /) -> None

      Add gate. Given gate is copied.


   .. py:method:: calculate_depth() -> int

      Get depth of circuit.


   .. py:method:: copy() -> Circuit

      Copy circuit. All the gates inside is copied.


   .. py:method:: gate_count() -> int

      Get property of `gate_count`.


   .. py:method:: gate_list() -> list[Gate]

      Get property of `gate_list`.


   .. py:method:: get(arg: int, /) -> Gate

      Get reference of i-th gate.


   .. py:method:: get_inverse() -> Circuit

      Get inverse of circuit. ALl the gates are newly created.


   .. py:method:: n_qubits() -> int

      Get property of `n_qubits`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to the StateVector. StateVector in args is directly updated.



.. py:function:: FusedSwap(arg0: int, arg1: int, arg2: int, /) -> Gate

   Generate general Gate class instance of FusedSwap.


.. py:class:: FusedSwapGate(arg: Gate, /)


       Specific class of fused swap gate, which swap qubits in $[\mathrm{qubit\_index1},\mathrm{qubit\_index1}+\mathrm{block\_size})$ and qubits in $[\mathrm{qubit\_index2},\mathrm{qubit\_index2}+\mathrm{block\_size})$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: block_size() -> int

      Get property `block_size`.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: qubit_index1() -> int

      Get property `qubit_index1`.


   .. py:method:: qubit_index2() -> int

      Get property `qubit_index2`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: Gate(arg: PauliRotationGate, /)


       General class of QuantumGate.
   .. note:: Downcast to requred to use gate-specific functions.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: GateType


   Bases: :py:obj:`enum.Enum`

   <attribute '__doc__' of 'GateType' objects>

   .. py:attribute:: CX
      :type: Any

      

   .. py:attribute:: CZ
      :type: Any

      

   .. py:attribute:: FusedSwap
      :type: Any

      

   .. py:attribute:: GlobalPhase
      :type: Any

      

   .. py:attribute:: H
      :type: Any

      

   .. py:attribute:: I
      :type: Any

      

   .. py:attribute:: OneQubitMatrix
      :type: Any

      

   .. py:attribute:: P0
      :type: Any

      

   .. py:attribute:: P1
      :type: Any

      

   .. py:attribute:: Pauli
      :type: Any

      

   .. py:attribute:: PauliRotation
      :type: Any

      

   .. py:attribute:: RX
      :type: Any

      

   .. py:attribute:: RY
      :type: Any

      

   .. py:attribute:: RZ
      :type: Any

      

   .. py:attribute:: S
      :type: Any

      

   .. py:attribute:: Sdag
      :type: Any

      

   .. py:attribute:: SqrtX
      :type: Any

      

   .. py:attribute:: SqrtXdag
      :type: Any

      

   .. py:attribute:: SqrtY
      :type: Any

      

   .. py:attribute:: SqrtYdag
      :type: Any

      

   .. py:attribute:: Swap
      :type: Any

      

   .. py:attribute:: T
      :type: Any

      

   .. py:attribute:: Tdag
      :type: Any

      

   .. py:attribute:: TwoQubitMatrix
      :type: Any

      

   .. py:attribute:: U1
      :type: Any

      

   .. py:attribute:: U2
      :type: Any

      

   .. py:attribute:: U3
      :type: Any

      

   .. py:attribute:: X
      :type: Any

      

   .. py:attribute:: Y
      :type: Any

      

   .. py:attribute:: Z
      :type: Any

      


.. py:function:: GlobalPhase(arg: float, /) -> Gate

   Generate general Gate class instance of GlobalPhase.


.. py:class:: GlobalPhaseGate(arg: Gate, /)


       Specific class of gate, which rotate global phase, represented as $e^{i\mathrm{phase}}I$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: phase() -> float

      Get `phase` property


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: H(arg: int, /) -> Gate

   Generate general Gate class instance of H.


.. py:class:: HGate(arg: Gate, /)


       Specific class of Hadamard gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: I() -> Gate

   Generate general Gate class instance of I.


.. py:class:: IGate(arg: Gate, /)


       Specific class of Pauli-I gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: InitializationSettings


       Wrapper class of Kokkos's InitializationSettings.
   .. note:: See details: https://kokkos.org/kokkos-core-wiki/API/core/initialize_finalize/InitializationSettings.html


   .. py:method:: get_device_id() -> int


   .. py:method:: get_disable_warnings() -> bool


   .. py:method:: get_map_device_id_by() -> str


   .. py:method:: get_num_threads() -> int


   .. py:method:: get_print_configuration() -> bool


   .. py:method:: get_tools_args() -> str


   .. py:method:: get_tools_help() -> bool


   .. py:method:: get_tools_libs() -> str


   .. py:method:: get_tune_internals() -> bool


   .. py:method:: has_device_id() -> bool


   .. py:method:: has_disable_warnings() -> bool


   .. py:method:: has_map_device_id_by() -> bool


   .. py:method:: has_num_threads() -> bool


   .. py:method:: has_print_configuration() -> bool


   .. py:method:: has_tools_args() -> bool


   .. py:method:: has_tools_help() -> bool


   .. py:method:: has_tools_libs() -> bool


   .. py:method:: has_tune_internals() -> bool


   .. py:method:: set_device_id(arg: int, /) -> InitializationSettings


   .. py:method:: set_disable_warnings(arg: bool, /) -> InitializationSettings


   .. py:method:: set_map_device_id_by(arg: str, /) -> InitializationSettings


   .. py:method:: set_num_threads(arg: int, /) -> InitializationSettings


   .. py:method:: set_print_configuration(arg: bool, /) -> InitializationSettings


   .. py:method:: set_tools_args(arg: str, /) -> InitializationSettings


   .. py:method:: set_tools_help(arg: bool, /) -> InitializationSettings


   .. py:method:: set_tools_libs(arg: str, /) -> InitializationSettings


   .. py:method:: set_tune_internals(arg: bool, /) -> InitializationSettings



.. py:class:: OneQubitMatrixGate(arg: Gate, /)


       Specific class of single-qubit dense matrix gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: matrix() -> list[list[complex]]


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: Operator(arg: int, /)


   None

   .. py:method:: add_operator(arg: PauliOperator, /) -> None


   .. py:method:: add_random_operator(operator_count: int, seed: Optional[int] = None) -> None


   .. py:method:: apply_to_state(arg: StateVector, /) -> None


   .. py:method:: get_dagger() -> Operator


   .. py:method:: get_expectation_value(arg: StateVector, /) -> complex


   .. py:method:: get_transition_amplitude(arg0: StateVector, arg1: StateVector, /) -> complex


   .. py:method:: is_hermitian() -> bool


   .. py:method:: n_qubits() -> int


   .. py:method:: optimize() -> None


   .. py:method:: terms() -> list[PauliOperator]


   .. py:method:: to_string() -> str



.. py:function:: P0(arg: int, /) -> Gate

   Generate general Gate class instance of P0.


.. py:class:: P0Gate(arg: Gate, /)


       Specific class of projection gate to $\ket{0}$.
   .. note:: This gate is not unitary.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: P1(arg: int, /) -> Gate

   Generate general Gate class instance of P1.


.. py:class:: P1Gate(arg: Gate, /)


       Specific class of projection gate to $\ket{1}$.
   .. note:: This gate is not unitary.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: Pauli(arg: PauliOperator, /) -> Gate

   Generate general Gate class instance of Pauli.


.. py:class:: PauliGate(arg: Gate, /)


       Specific class of multi-qubit pauli gate, which applies single-qubit Pauli gate to each of qubit.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: PauliOperator(bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0)


   None

   .. py:method:: add_single_pauli(arg0: int, arg1: int, /) -> None


   .. py:method:: apply_to_state(arg: StateVector, /) -> None


   .. py:method:: change_coef(arg: complex, /) -> None


   .. py:method:: get_XZ_mask_representation() -> tuple[int, int]


   .. py:method:: get_coef() -> complex


   .. py:method:: get_dagger() -> PauliOperator


   .. py:method:: get_expectation_value(arg: StateVector, /) -> complex


   .. py:method:: get_pauli_id_list() -> list[int]


   .. py:method:: get_pauli_string() -> str


   .. py:method:: get_qubit_count() -> int


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: get_transition_amplitude(arg0: StateVector, arg1: StateVector, /) -> complex



.. py:function:: PauliRotation(arg0: PauliOperator, arg1: float, /) -> Gate

   Generate general Gate class instance of PauliRotation.


.. py:class:: PauliRotationGate(arg: Gate, /)


       Specific class of multi-qubit pauli-rotation gate, represented as $e^{-i\frac{\mathrm{angle}}{2}P}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: RX(arg0: int, arg1: float, /) -> Gate

   Generate general Gate class instance of RX.


.. py:class:: RXGate(arg: Gate, /)


       Specific class of X rotation gate, represented as $e^{-i\frac{\mathrm{angle}}{2}X}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: angle() -> float

      Get `angle` property.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: RY(arg0: int, arg1: float, /) -> Gate

   Generate general Gate class instance of RY.


.. py:class:: RYGate(arg: Gate, /)


       Specific class of Y rotation gate, represented as $e^{-i\frac{\mathrm{angle}}{2}Y}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: angle() -> float

      Get `angle` property.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: RZ(arg0: int, arg1: float, /) -> Gate

   Generate general Gate class instance of RZ.


.. py:class:: RZGate(arg: Gate, /)


       Specific class of Z rotation gate, represented as $e^{-i\frac{\mathrm{angle}}{2}Z}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: angle() -> float

      Get `angle` property.


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: S(arg: int, /) -> Gate

   Generate general Gate class instance of S.


.. py:class:: SGate(arg: Gate, /)


       Specific class of S gate, represented as $\begin{bmatrix}
       1 & 0\\
       0 & i
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: Sdag(arg: int, /) -> Gate

   Generate general Gate class instance of Sdag.


.. py:class:: SdagGate(arg: Gate, /)


       Specific class of inverse of S gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: SqrtX(arg: int, /) -> Gate

   Generate general Gate class instance of SqrtX.


.. py:class:: SqrtXGate(arg: Gate, /)


       Specific class of sqrt(X) gate, represented as $\begin{bmatrix}
       1+i & 1-i\\
       1-i & 1+i
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: SqrtXdag(arg: int, /) -> Gate

   Generate general Gate class instance of SqrtXdag.


.. py:class:: SqrtXdagGate(arg: Gate, /)


       Specific class of inverse of sqrt(X) gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: SqrtY(arg: int, /) -> Gate

   Generate general Gate class instance of SqrtY.


.. py:class:: SqrtYGate(arg: Gate, /)


       Specific class of sqrt(Y) gate, represented as $\begin{bmatrix}
       1+i & -1-i \\
       1+i & 1+i
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: SqrtYdag(arg: int, /) -> Gate

   Generate general Gate class instance of SqrtYdag.


.. py:class:: SqrtYdagGate(arg: Gate, /)


       Specific class of inverse of sqrt(Y) gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: StateVector(arg: StateVector)


       Vector representation of quantum state.
   .. note:: Qubit index is start from 0. If the amplitudes of $\ket{b_{n-1}\dots b_0}$ is $b_i$, the state is $\sum_i b_i 2^i$.


   .. py:method:: Haar_random_state(seed: Optional[int] = None) -> StateVector

      Constructing state vector with Haar random state. If seed is not specified, the value from random device is used.


   .. py:method:: add_state_vector(arg: StateVector, /) -> None

      Add other state vector and make superposition. $\ket{\mathrm{this}} \leftarrow \ket{\mathrm{this}} + \ket{\mathrm{state}}$.


   .. py:method:: add_state_vector_with_coef(arg0: complex, arg1: StateVector, /) -> None

      add other state vector with multiplying the coef and make superposition. $\ket{\mathrm{this}}\leftarrow\ket{\mathrm{this}}+\mathrm{coef}\ket{\mathrm{state}}$.


   .. py:method:: amplitudes() -> list[complex]

      Get all amplitudes with as `list[complex]`.


   .. py:method:: dim() -> int

      Get dimension of the vector ($=2^\mathrm{n\_qubits}$).


   .. py:method:: get_amplitude_at_index(arg: int, /) -> complex

              Get amplitude at one index.
      .. note:: If you want to get all amplitudes, you should use `StateVector::amplitudes()`.



   .. py:method:: get_entropy() -> float

      Get the entropy of the vector.


   .. py:method:: get_marginal_probability(arg: list[int], /) -> float

      Get the marginal probability to observe as specified. Specify the result as n-length list. `0` and `1` represent the qubit is observed and get the value. `2` represents the qubit is not observed.


   .. py:method:: get_squared_norm() -> float

      Get squared norm of the state. $\braket{\psi|\psi}$.


   .. py:method:: get_zero_probability(arg: int, /) -> float

      Get the probability to observe $\ket{0}$ at specified index.


   .. py:method:: load(arg: list[complex], /) -> None

      Load amplitudes of `list[int]` with `dim` length.


   .. py:method:: multiply_coef(arg: complex, /) -> None

      Multiply coef. $\ket{\mathrm{this}}\leftarrow\mathrm{coef}\ket{\mathrm{this}}$.


   .. py:method:: n_qubits() -> int

      Get num of qubits.


   .. py:method:: normalize() -> None

      Normalize state (let $\braket{\psi|\psi} = 1$ by multiplying coef).


   .. py:method:: sampling(sampling_count: int, seed: Optional[int] = None) -> list[int]

      Sampling specified times. Result is `list[int]` with the `sampling_count` length.


   .. py:method:: set_amplitude_at_index(arg0: int, arg1: complex, /) -> None

      Manually set amplitude at one index.


   .. py:method:: set_computational_basis(arg: int, /) -> None

      Initialize with computational basis \ket{\mathrm{basis}}.


   .. py:method:: set_zero_norm_state() -> None

      Initialize with 0 (null vector).


   .. py:method:: set_zero_state() -> None

      Initialize with computational basis $\ket{00\dots0}$.


   .. py:method:: to_string() -> str

      Information as `str`.



.. py:function:: Swap(arg0: int, arg1: int, /) -> Gate

   Generate general Gate class instance of Swap.


.. py:class:: SwapGate(arg: Gate, /)


       Specific class of two-qubit swap gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target1() -> int

      Get property `target1`.


   .. py:method:: target2() -> int

      Get property `target2`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: T(arg: int, /) -> Gate

   Generate general Gate class instance of T.


.. py:class:: TGate(arg: Gate, /)


       Specific class of T gate, represented as $\begin{bmatrix}
       1 & 0\\
       0 & e^{i\pi/4}
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: Tdag(arg: int, /) -> Gate

   Generate general Gate class instance of Tdag.


.. py:class:: TdagGate(arg: Gate, /)


       Specific class of inverse of T gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:class:: TwoQubitMatrixGate(arg: Gate, /)


       Specific class of double-qubit dense matrix gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: matrix() -> None

      Get property `matrix`.


   .. py:method:: target1() -> int

      Get property `target1`.


   .. py:method:: target2() -> int

      Get property `target2`.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: U1(arg0: int, arg1: float, /) -> Gate

   Generate general Gate class instance of U1.


.. py:class:: U1Gate(arg: Gate, /)


       Specific class of IBMQ's U1 Gate, which is a rotation abount Z-axis, represented as $\begin{bmatrix}
       1 & 0\\
       0 & e^{i\lambda}
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: lambda_() -> float

      Get `lambda` property.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: U2(arg0: int, arg1: float, arg2: float, /) -> Gate

   Generate general Gate class instance of U2.


.. py:class:: U2Gate(arg: Gate, /)


       Specific class of IBMQ's U2 Gate, which is a rotation about X+Z-axis, represented as $\frac{1}{\sqrt{2}} \begin{bmatrix}1 & -e^{-i\lambda}\\
       e^{i\phi} & e^{i(\phi+\lambda)}
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: lambda_() -> float

      Get `lambda` property.


   .. py:method:: phi() -> float

      Get `phi` property.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: U3(arg0: int, arg1: float, arg2: float, arg3: float, /) -> Gate

   Generate general Gate class instance of U3.


.. py:class:: U3Gate(arg: Gate, /)


       Specific class of IBMQ's U3 Gate, which is a rotation abount 3 axis, represented as $\begin{bmatrix}
       \cos \frac{\theta}{2} & -e^{i\lambda}\sin\frac{\theta}{2}\\
       e^{i\phi}\sin\frac{\theta}{2} & e^{i(\phi+\lambda)}\cos\frac{\theta}{2}
       \end{bmatrix}$.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: lambda_() -> float

      Get `lambda` property.


   .. py:method:: phi() -> float

      Get `phi` property.


   .. py:method:: theta() -> float

      Get `theta` property.


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: X(arg: int, /) -> Gate

   Generate general Gate class instance of X.


.. py:class:: XGate(arg: Gate, /)


       Specific class of Pauli-X gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: Y(arg: int, /) -> Gate

   Generate general Gate class instance of Y.


.. py:class:: YGate(arg: Gate, /)


       Specific class of Pauli-Y gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: Z(arg: int, /) -> Gate

   Generate general Gate class instance of Z.


.. py:class:: ZGate(arg: Gate, /)


       Specific class of Pauli-Z gate.
   .. note:: Upcast is required to use gate-general functions (ex: add to Circuit).


   .. py:method:: copy() -> Gate

      Copy gate as `Gate` type.


   .. py:method:: gate_type() -> GateType

      Get gate type as `GateType` enum.


   .. py:method:: get_control_qubit_list() -> list[int]

      Get control qubits as `list[int]`.


   .. py:method:: get_inverse() -> Gate

      Generate inverse gate as `Gate` type. If not exists, return None.


   .. py:method:: get_target_qubit_list() -> list[int]

      Get target qubits as `list[int]`. **Control qubits is not included.**


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None

      Apply gate to `state_vector`. `state_vector` in args is directly updated.



.. py:function:: finalize() -> None

   Terminate the Kokkos execution environment. Release the resources.


.. py:function:: initialize(settings: InitializationSettings = ...) -> None

   **You must call this before any scaluq function.** Initialize the Kokkos execution environment.


.. py:function:: is_finalized() -> bool

   Return true if `finalize()` is already called.


.. py:function:: is_initialized() -> bool

   Return true if `initialize()` is already called.


