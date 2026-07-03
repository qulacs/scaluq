from .scaluq_core import *
try:
    from . import gate
except AttributeError:
    gate = None

import inspect

_PRECISIONS = ['f64', 'f32', 'f16', 'bf16']
_SPACES = ['default', 'host', 'host_serial']
_available_precisions = [prec for prec in _PRECISIONS if precision_available(prec)]
_DEFAULT_PRECISION = _available_precisions[0]

_modules: dict[tuple[str, str], object] = {}
for _space in _SPACES:
    _space_mod = getattr(scaluq_core, _space, None)
    if _space_mod is None:
        continue
    for _prec in _available_precisions:
        _prec_mod = getattr(_space_mod, _prec, None)
        if _prec_mod is None:
            continue
        _modules[(_space, _prec)] = _prec_mod
_mod_id_to_key = {id(mod): key for key, mod in _modules.items()}

def _get_module(precision, space):
    if precision is None:
        precision = _DEFAULT_PRECISION
    try:
        return _modules[(space, precision)]
    except KeyError:
        if space not in _SPACES:
            raise ValueError(f"Execution space {space} is not supported.")
        raise ValueError(f"Precision {precision} is not available.")

def _key_for_object(obj):
    mod = inspect.getmodule(obj)
    return _mod_id_to_key.get(id(mod)) if mod is not None else None

class StateVector:
    UNMEASURED = _get_module(_DEFAULT_PRECISION, 'default').StateVector.UNMEASURED

    def __new__(cls, n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVector(n_qubits)

    @staticmethod
    def Haar_random_state(n_qubits, seed=None, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVector.Haar_random_state(n_qubits, seed)

    @staticmethod
    def uninitialized_state(n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVector.uninitialized_state(n_qubits)
    
    @staticmethod
    def inner_product(state1, state2):
        if type(state1) is not type(state2):
            raise ValueError("State vectors must be of the same type for inner product.")
        key = _key_for_object(state1)
        if key is None:
            raise ValueError(f"Unsupported state vector type: {type(state1)}")
        return _modules[key].StateVector.inner_product(state1, state2)

class StateVectorBatched:
    def __new__(cls, batch_size, n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVectorBatched(batch_size, n_qubits)

    @staticmethod
    def Haar_random_state(batch_size, n_qubits, set_same_state, seed=None, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVectorBatched.Haar_random_state(batch_size, n_qubits, set_same_state, seed)

    @staticmethod
    def uninitialized_state(batch_size, n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).StateVectorBatched.uninitialized_state(batch_size, n_qubits)

class DensityMatrix:
    UNMEASURED = _get_module(_DEFAULT_PRECISION, 'default').DensityMatrix.UNMEASURED

    def __new__(cls, n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).DensityMatrix(n_qubits)

    @staticmethod
    def Haar_random_state(n_qubits, seed=None, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).DensityMatrix.Haar_random_state(n_qubits, seed)

    @staticmethod
    def uninitialized_state(n_qubits, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).DensityMatrix.uninitialized_state(n_qubits)

def Gate(gate):
    key = _key_for_object(gate)
    if key is None:
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    return _modules[('default', key[1])].Gate(gate)

def ParamGate(gate):
    key = _key_for_object(gate)
    if key is None:
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    return _modules[('default', key[1])].ParamGate(gate)

def _make_gate_wrapper(name, is_param, is_space_specific):
    def caster(gate, space='default'):
        key = _key_for_object(gate)
        if key is None:
            raise ValueError(f"Unsupported gate type: {type(gate)}")
        if is_param:
            gate_cls = _get_module(key[1], 'default').ParamGate
        else:
            gate_cls = _get_module(key[1], 'default').Gate
        spec_cls = getattr(_get_module(key[1], space if is_space_specific else 'default'), name, None)
        if spec_cls is None:
            raise ValueError(f"{name} is not available in space={target_space}")
        if type(gate) is spec_cls:
            return gate
        if type(gate) is gate_cls:
            return spec_cls(gate)
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    caster.__name__ = name
    caster.__qualname__ = name
    return caster

if len(_available_precisions) > 0:
    _one_prec = _available_precisions[0]
    _default_mod = _get_module(_one_prec, 'default')
    for _name in dir(_default_mod):
        if _name.startswith('_'):
            continue
        if not _name.endswith('Gate'):
            continue
        if _name == 'Gate':
            continue
        globals()[_name] = _make_gate_wrapper(_name, is_param=(_name.startswith('Param')), is_space_specific=(_name in ['SparseMatrixGate', 'DenseMatrixGate']))

def merge_gate(gate1, gate2):
    key1 = _key_for_object(gate1)
    key2 = _key_for_object(gate2)
    if key1 is None:
        raise ValueError(f"Unsupported gate type: {type(gate1)}")
    if key2 is None:
        raise ValueError(f"Unsupported gate type: {type(gate2)}")
    if key1 != key2:
        raise ValueError("Gates must be from the same precision and execution space.")
    mod = _modules[key1]
    if not hasattr(mod, 'merge_gate'):
        space, precision = key1
        raise ValueError(f"merge_gate is not available in space={space} with precision={precision}")
    return mod.merge_gate(gate1, gate2)

class PauliOperator:
    @staticmethod
    def from_targets_and_pauli_ids(target_qubit_list, pauli_id_list, coef=1.0, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').PauliOperator(target_qubit_list, pauli_id_list, coef=coef)

    @staticmethod
    def from_pauli_string(pauli_string, coef=1.0, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').PauliOperator(pauli_string, coef=coef)
    
    @staticmethod
    def from_pauli_id_par_qubit(pauli_id_par_qubit, coef=1.0, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').PauliOperator(pauli_id_par_qubit, coef=coef)
    
    @staticmethod
    def from_XZ_mask(bit_flip_mask, phase_flip_mask, coef=1.0, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').PauliOperator(bit_flip_mask, phase_flip_mask, coef=coef)

    def __new__(cls, pauli_string="", coef=1.0, precision=_DEFAULT_PRECISION):
        return PauliOperator.from_pauli_string(pauli_string, coef=coef, precision=precision)


class Operator:
    def __new__(cls, terms, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).Operator(terms)

class OperatorBatched:
    def __new__(cls, terms, precision=_DEFAULT_PRECISION, space='default'):
        return _get_module(precision, space).OperatorBatched(terms)

class Circuit:
    def __new__(cls, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').Circuit()

class qasm2:
    @staticmethod
    def loads(source, precision=_DEFAULT_PRECISION):
        return _get_module(precision, 'default').qasm2.loads(source)

    @staticmethod
    def dumps(circuit, n_qubits=None):
        key = _key_for_object(circuit)
        if key is None:
            raise ValueError(f"Cannot infer precision from circuit type: {type(circuit)}")
        prec_mod = _get_module(key[1], 'default')
        if n_qubits is None:
            return prec_mod.qasm2.dumps(circuit)
        return prec_mod.qasm2.dumps(circuit, n_qubits)
