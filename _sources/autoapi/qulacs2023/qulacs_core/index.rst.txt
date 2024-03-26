:py:mod:`qulacs2023.qulacs_core`
================================

.. py:module:: qulacs2023.qulacs_core


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   qulacs2023.qulacs_core.CXGate
   qulacs2023.qulacs_core.CZGate
   qulacs2023.qulacs_core.Circuit
   qulacs2023.qulacs_core.FusedSwapGate
   qulacs2023.qulacs_core.Gate
   qulacs2023.qulacs_core.GateType
   qulacs2023.qulacs_core.GlobalPhaseGate
   qulacs2023.qulacs_core.HGate
   qulacs2023.qulacs_core.IGate
   qulacs2023.qulacs_core.InitializationSettings
   qulacs2023.qulacs_core.OneQubitMatrixGate
   qulacs2023.qulacs_core.Operator
   qulacs2023.qulacs_core.P0Gate
   qulacs2023.qulacs_core.P1Gate
   qulacs2023.qulacs_core.PauliGate
   qulacs2023.qulacs_core.PauliOperator
   qulacs2023.qulacs_core.PauliRotationGate
   qulacs2023.qulacs_core.RXGate
   qulacs2023.qulacs_core.RYGate
   qulacs2023.qulacs_core.RZGate
   qulacs2023.qulacs_core.SGate
   qulacs2023.qulacs_core.SdagGate
   qulacs2023.qulacs_core.SqrtXGate
   qulacs2023.qulacs_core.SqrtXdagGate
   qulacs2023.qulacs_core.SqrtYGate
   qulacs2023.qulacs_core.SqrtYdagGate
   qulacs2023.qulacs_core.StateVector
   qulacs2023.qulacs_core.SwapGate
   qulacs2023.qulacs_core.TGate
   qulacs2023.qulacs_core.TdagGate
   qulacs2023.qulacs_core.TwoQubitMatrixGate
   qulacs2023.qulacs_core.U1Gate
   qulacs2023.qulacs_core.U2Gate
   qulacs2023.qulacs_core.U3Gate
   qulacs2023.qulacs_core.XGate
   qulacs2023.qulacs_core.YGate
   qulacs2023.qulacs_core.ZGate



Functions
~~~~~~~~~

.. autoapisummary::

   qulacs2023.qulacs_core.CNot
   qulacs2023.qulacs_core.CX
   qulacs2023.qulacs_core.CZ
   qulacs2023.qulacs_core.FusedSwap
   qulacs2023.qulacs_core.GlobalPhase
   qulacs2023.qulacs_core.H
   qulacs2023.qulacs_core.I
   qulacs2023.qulacs_core.P0
   qulacs2023.qulacs_core.P1
   qulacs2023.qulacs_core.Pauli
   qulacs2023.qulacs_core.PauliRotation
   qulacs2023.qulacs_core.RX
   qulacs2023.qulacs_core.RY
   qulacs2023.qulacs_core.RZ
   qulacs2023.qulacs_core.S
   qulacs2023.qulacs_core.Sdag
   qulacs2023.qulacs_core.SqrtX
   qulacs2023.qulacs_core.SqrtXdag
   qulacs2023.qulacs_core.SqrtY
   qulacs2023.qulacs_core.SqrtYdag
   qulacs2023.qulacs_core.Swap
   qulacs2023.qulacs_core.T
   qulacs2023.qulacs_core.Tdag
   qulacs2023.qulacs_core.U1
   qulacs2023.qulacs_core.U2
   qulacs2023.qulacs_core.U3
   qulacs2023.qulacs_core.X
   qulacs2023.qulacs_core.Y
   qulacs2023.qulacs_core.Z
   qulacs2023.qulacs_core.finalize
   qulacs2023.qulacs_core.initialize



.. py:function:: CNot(arg0: int, arg1: int, /) -> Gate


.. py:function:: CX(arg0: int, arg1: int, /) -> Gate


.. py:class:: CXGate(arg: Gate, /)


   None

   .. py:method:: control() -> int


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: CZ(arg0: int, arg1: int, /) -> Gate


.. py:class:: CZGate(arg: Gate, /)


   None

   .. py:method:: control() -> int


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:class:: Circuit(arg: int, /)


   None

   .. py:method:: add_circuit(arg: Circuit, /) -> None


   .. py:method:: add_gate(arg: Gate, /) -> None


   .. py:method:: calculate_depth() -> int


   .. py:method:: copy() -> Circuit


   .. py:method:: gate_count() -> int


   .. py:method:: gate_list() -> list[Gate]


   .. py:method:: get(arg: int, /) -> Gate


   .. py:method:: get_inverse() -> Circuit


   .. py:method:: n_qubits() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: FusedSwap(arg0: int, arg1: int, arg2: int, /) -> Gate


.. py:class:: FusedSwapGate(arg: Gate, /)


   None

   .. py:method:: block_size() -> int


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: qubit_index1() -> int


   .. py:method:: qubit_index2() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:class:: Gate(arg: PauliRotationGate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



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


.. py:class:: GlobalPhaseGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: phase() -> float


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: H(arg: int, /) -> Gate


.. py:class:: HGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: I() -> Gate


.. py:class:: IGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



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


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: matrix(**kwargs)

      matrix(self) -> std::array<std::array<Kokkos::complex<double>, 2ul>, 2ul>


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



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


.. py:class:: P0Gate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: P1(arg: int, /) -> Gate


.. py:class:: P1Gate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: Pauli(arg: PauliOperator, /) -> Gate


.. py:class:: PauliGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



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


.. py:class:: PauliRotationGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: RX(arg0: int, arg1: float, /) -> Gate


.. py:class:: RXGate(arg: Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: RY(arg0: int, arg1: float, /) -> Gate


.. py:class:: RYGate(arg: Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: RZ(arg0: int, arg1: float, /) -> Gate


.. py:class:: RZGate(arg: Gate, /)


   None

   .. py:method:: angle() -> float


   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: S(arg: int, /) -> Gate


.. py:class:: SGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: Sdag(arg: int, /) -> Gate


.. py:class:: SdagGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: SqrtX(arg: int, /) -> Gate


.. py:class:: SqrtXGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: SqrtXdag(arg: int, /) -> Gate


.. py:class:: SqrtXdagGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: SqrtY(arg: int, /) -> Gate


.. py:class:: SqrtYGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: SqrtYdag(arg: int, /) -> Gate


.. py:class:: SqrtYdagGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:class:: StateVector(arg: StateVector)


   None

   .. py:method:: Haar_random_state(seed: Optional[int] = None) -> StateVector


   .. py:method:: add_state_vector(arg: StateVector, /) -> None


   .. py:method:: add_state_vector_with_coef(arg0: complex, arg1: StateVector, /) -> None


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



.. py:function:: Swap(arg0: int, arg1: int, /) -> Gate


.. py:class:: SwapGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target1() -> int


   .. py:method:: target2() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: T(arg: int, /) -> Gate


.. py:class:: TGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: Tdag(arg: int, /) -> Gate


.. py:class:: TdagGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:class:: TwoQubitMatrixGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: matrix() -> None


   .. py:method:: target1() -> int


   .. py:method:: target2() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: U1(arg0: int, arg1: float, /) -> Gate


.. py:class:: U1Gate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: U2(arg0: int, arg1: float, arg2: float, /) -> Gate


.. py:class:: U2Gate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: phi() -> float


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: U3(arg0: int, arg1: float, arg2: float, arg3: float, /) -> Gate


.. py:class:: U3Gate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: lambda_() -> float


   .. py:method:: phi() -> float


   .. py:method:: theta() -> float


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: X(arg: int, /) -> Gate


.. py:class:: XGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: Y(arg: int, /) -> Gate


.. py:class:: YGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: Z(arg: int, /) -> Gate


.. py:class:: ZGate(arg: Gate, /)


   None

   .. py:method:: copy() -> Gate


   .. py:method:: gate_type() -> GateType


   .. py:method:: get_control_qubit_list() -> list[int]


   .. py:method:: get_inverse() -> Gate


   .. py:method:: get_target_qubit_list() -> list[int]


   .. py:method:: target() -> int


   .. py:method:: update_quantum_state(arg: StateVector, /) -> None



.. py:function:: finalize() -> None


.. py:function:: initialize(settings: InitializationSettings = ...) -> None


