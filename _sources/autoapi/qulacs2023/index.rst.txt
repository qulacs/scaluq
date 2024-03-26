:py:mod:`qulacs2023`
====================

.. py:module:: qulacs2023


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   qulacs_core/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   qulacs2023.CXGate
   qulacs2023.CZGate
   qulacs2023.Circuit
   qulacs2023.FusedSwapGate
   qulacs2023.Gate
   qulacs2023.GateType
   qulacs2023.GlobalPhaseGate
   qulacs2023.HGate
   qulacs2023.IGate
   qulacs2023.InitializationSettings
   qulacs2023.OneQubitMatrixGate
   qulacs2023.Operator
   qulacs2023.P0Gate
   qulacs2023.P1Gate
   qulacs2023.PauliGate
   qulacs2023.PauliOperator
   qulacs2023.PauliRotationGate
   qulacs2023.RXGate
   qulacs2023.RYGate
   qulacs2023.RZGate
   qulacs2023.SGate
   qulacs2023.SdagGate
   qulacs2023.SqrtXGate
   qulacs2023.SqrtXdagGate
   qulacs2023.SqrtYGate
   qulacs2023.SqrtYdagGate
   qulacs2023.StateVector
   qulacs2023.SwapGate
   qulacs2023.TGate
   qulacs2023.TdagGate
   qulacs2023.TwoQubitMatrixGate
   qulacs2023.U1Gate
   qulacs2023.U2Gate
   qulacs2023.U3Gate
   qulacs2023.XGate
   qulacs2023.YGate
   qulacs2023.ZGate



Functions
~~~~~~~~~

.. autoapisummary::

   qulacs2023.CNot
   qulacs2023.CX
   qulacs2023.CZ
   qulacs2023.FusedSwap
   qulacs2023.GlobalPhase
   qulacs2023.H
   qulacs2023.I
   qulacs2023.P0
   qulacs2023.P1
   qulacs2023.Pauli
   qulacs2023.PauliRotation
   qulacs2023.RX
   qulacs2023.RY
   qulacs2023.RZ
   qulacs2023.S
   qulacs2023.Sdag
   qulacs2023.SqrtX
   qulacs2023.SqrtXdag
   qulacs2023.SqrtY
   qulacs2023.SqrtYdag
   qulacs2023.Swap
   qulacs2023.T
   qulacs2023.Tdag
   qulacs2023.U1
   qulacs2023.U2
   qulacs2023.U3
   qulacs2023.X
   qulacs2023.Y
   qulacs2023.Z
   qulacs2023.finalize
   qulacs2023.initialize



.. py:function:: CNot(arg0: int, arg1: int, /) -> qulacs_core.Gate


.. py:function:: CX(arg0: int, arg1: int, /) -> qulacs_core.Gate


.. py:class:: CXGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: control() -> int


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: CZ(arg0: int, arg1: int, /) -> qulacs_core.Gate


.. py:class:: CZGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: control() -> int


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: Circuit(arg: int, /)


   None

   .. py:method:: add_circuit(arg: qulacs_core.Circuit, /) -> None


   .. py:method:: add_gate(arg: qulacs_core.Gate, /) -> None


   .. py:method:: calculate_depth() -> int


   .. py:method:: copy() -> qulacs_core.Circuit


   .. py:method:: gate_count() -> int


   .. py:method:: gate_list() -> list[qulacs_core.Gate]


   .. py:method:: get(arg: int, /) -> qulacs_core.Gate


   .. py:method:: get_inverse() -> qulacs_core.Circuit


   .. py:method:: n_qubits() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: FusedSwap(arg0: int, arg1: int, arg2: int, /) -> qulacs_core.Gate


.. py:class:: FusedSwapGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: block_size() -> int


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: qubit_index1() -> int


   .. py:method:: qubit_index2() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: Gate(arg: qulacs_core.PauliRotationGate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



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

      


.. py:function:: GlobalPhase(arg: float, /) -> qulacs_core.Gate


.. py:class:: GlobalPhaseGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: phase() -> float


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: H(arg: int, /) -> qulacs_core.Gate


.. py:class:: HGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: I() -> qulacs_core.Gate


.. py:class:: IGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: InitializationSettings


   None

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


   .. py:method:: set_device_id(arg: int, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_disable_warnings(arg: bool, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_map_device_id_by(arg: str, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_num_threads(arg: int, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_print_configuration(arg: bool, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_tools_args(arg: str, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_tools_help(arg: bool, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_tools_libs(arg: str, /) -> qulacs_core.InitializationSettings


   .. py:method:: set_tune_internals(arg: bool, /) -> qulacs_core.InitializationSettings



.. py:class:: OneQubitMatrixGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: matrix(**kwargs)

      matrix(self) -> std::array<std::array<Kokkos::complex<double>, 2ul>, 2ul>


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: Operator(arg: int, /)


   None

   .. py:method:: add_operator(arg: qulacs_core.PauliOperator, /) -> None


   .. py:method:: add_random_operator(operator_count: int, seed: Optional[int] = None) -> None


   .. py:method:: apply_to_state(arg: qulacs_core.StateVector, /) -> None


   .. py:method:: get_dagger() -> qulacs_core.Operator


   .. py:method:: get_expectation_value(arg: qulacs_core.StateVector, /) -> complex


   .. py:method:: get_transition_amplitude(arg0: qulacs_core.StateVector, arg1: qulacs_core.StateVector, /) -> complex


   .. py:method:: is_hermitian() -> bool


   .. py:method:: n_qubits() -> int


   .. py:method:: optimize() -> None


   .. py:method:: terms() -> list[qulacs_core.PauliOperator]


   .. py:method:: to_string() -> str



.. py:function:: P0(arg: int, /) -> qulacs_core.Gate


.. py:class:: P0Gate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: P1(arg: int, /) -> qulacs_core.Gate


.. py:class:: P1Gate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: Pauli(arg: qulacs_core.PauliOperator, /) -> qulacs_core.Gate


.. py:class:: PauliGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: PauliOperator(bit_flip_mask: int, phase_flip_mask: int, coef: complex = 1.0)


   None

   .. py:method:: add_single_pauli(arg0: int, arg1: int, /) -> None


   .. py:method:: apply_to_state(arg: qulacs_core.StateVector, /) -> None


   .. py:method:: change_coef(arg: complex, /) -> None


   .. py:method:: get_XZ_mask_representation() -> tuple[int, int]


   .. py:method:: get_coef() -> complex


   .. py:method:: get_dagger() -> qulacs_core.PauliOperator


   .. py:method:: get_expectation_value(arg: qulacs_core.StateVector, /) -> complex


   .. py:method:: get_pauli_id_list() -> list[int]


   .. py:method:: get_pauli_string() -> str


   .. py:method:: get_qubit_count() -> int


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: get_transition_amplitude(arg0: qulacs_core.StateVector, arg1: qulacs_core.StateVector, /) -> complex



.. py:function:: PauliRotation(arg0: qulacs_core.PauliOperator, arg1: float, /) -> qulacs_core.Gate


.. py:class:: PauliRotationGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: RX(arg0: int, arg1: float, /) -> qulacs_core.Gate


.. py:class:: RXGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: RY(arg0: int, arg1: float, /) -> qulacs_core.Gate


.. py:class:: RYGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: RZ(arg0: int, arg1: float, /) -> qulacs_core.Gate


.. py:class:: RZGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: S(arg: int, /) -> qulacs_core.Gate


.. py:class:: SGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: Sdag(arg: int, /) -> qulacs_core.Gate


.. py:class:: SdagGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: SqrtX(arg: int, /) -> qulacs_core.Gate


.. py:class:: SqrtXGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: SqrtXdag(arg: int, /) -> qulacs_core.Gate


.. py:class:: SqrtXdagGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: SqrtY(arg: int, /) -> qulacs_core.Gate


.. py:class:: SqrtYGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: SqrtYdag(arg: int, /) -> qulacs_core.Gate


.. py:class:: SqrtYdagGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: StateVector(arg: qulacs_core.StateVector)


   None

   .. py:method:: Haar_random_state(seed: Optional[int] = None) -> qulacs_core.StateVector


   .. py:method:: add_state_vector(arg: qulacs_core.StateVector, /) -> None


   .. py:method:: add_state_vector_with_coef(arg0: complex, arg1: qulacs_core.StateVector, /) -> None


   .. py:method:: amplitudes() -> list[complex]


   .. py:method:: dim() -> int


   .. py:method:: get_amplitude_at_index(arg: int, /) -> complex


   .. py:method:: get_entropy() -> float


   .. py:method:: get_marginal_probability(arg: list[int], /) -> float


   .. py:method:: get_squared_norm() -> float


   .. py:method:: get_zero_probability(arg: int, /) -> float


   .. py:method:: load(arg: list[complex], /) -> None


   .. py:method:: multiply_coef(arg: complex, /) -> None


   .. py:method:: n_qubits() -> int


   .. py:method:: normalize() -> None


   .. py:method:: sampling(sampling_count: int, seed: Optional[int] = None) -> list[int]


   .. py:method:: set_amplitude_at_index(arg0: int, arg1: complex, /) -> None


   .. py:method:: set_computational_basis(arg: int, /) -> None


   .. py:method:: set_zero_norm_state() -> None


   .. py:method:: set_zero_state() -> None


   .. py:method:: to_string() -> str



.. py:function:: Swap(arg0: int, arg1: int, /) -> qulacs_core.Gate


.. py:class:: SwapGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target1() -> int


   .. py:method:: target2() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: T(arg: int, /) -> qulacs_core.Gate


.. py:class:: TGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: Tdag(arg: int, /) -> qulacs_core.Gate


.. py:class:: TdagGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:class:: TwoQubitMatrixGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: matrix() -> None


   .. py:method:: target1() -> int


   .. py:method:: target2() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: U1(arg0: int, arg1: float, /) -> qulacs_core.Gate


.. py:class:: U1Gate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: U2(arg0: int, arg1: float, arg2: float, /) -> qulacs_core.Gate


.. py:class:: U2Gate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: phi() -> float


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: U3(arg0: int, arg1: float, arg2: float, arg3: float, /) -> qulacs_core.Gate


.. py:class:: U3Gate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: phi() -> float


   .. py:method:: theta() -> float


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: X(arg: int, /) -> qulacs_core.Gate


.. py:class:: XGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: Y(arg: int, /) -> qulacs_core.Gate


.. py:class:: YGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: Z(arg: int, /) -> qulacs_core.Gate


.. py:class:: ZGate(arg: qulacs_core.Gate, /)


   None

   .. py:method:: copy() -> qulacs_core.Gate


   .. py:method:: gate_type() -> qulacs_core.GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> qulacs_core.Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: qulacs_core.StateVector, /) -> None



.. py:function:: finalize() -> None


.. py:function:: initialize(settings: qulacs_core.InitializationSettings = ...) -> None


