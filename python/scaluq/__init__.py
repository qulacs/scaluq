from .scaluq_core import *
try:
    from . import gate
except AttributeError:
    gate = None

import inspect

_PRECISIONS = ['f64', 'f32', 'f16', 'bf16']
_SPACES = ['default', 'host', 'host_serial']
_available_precisions = [prec for prec in _PRECISIONS if precision_available(prec)]

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
    try:
        return _modules[(space, precision)]
    except KeyError:
        if space not in _SPACES:
            raise ValueError(f"Execution space {space} is not supported.")
        raise ValueError(f"Precision {precision} is not available.")

def _key_for_gate(gate):
    mod = inspect.getmodule(gate)
    return _mod_id_to_key.get(id(mod)) if mod is not None else None

class StateVector:
    def __new__(cls, n_qubits, precision='f64', space='default'):
        return _get_module(precision, space).StateVector(n_qubits)

    @staticmethod
    def Haar_random_state(n_qubits, seed=None, precision='f64', space='default'):
        return _get_module(precision, space).StateVector.Haar_random_state(n_qubits, seed)

    @staticmethod
    def uninitialized_state(n_qubits, precision='f64', space='default'):
        return _get_module(precision, space).StateVector.uninitialized_state(n_qubits)

class StateVectorBatched:
    def __new__(cls, batch_size, n_qubits, precision='f64', space='default'):
        return _get_module(precision, space).StateVectorBatched(batch_size, n_qubits)

    @staticmethod
    def Haar_random_state(batch_size, n_qubits, seed=None, precision='f64', space='default'):
        return _get_module(precision, space).StateVectorBatched.Haar_random_state(batch_size, n_qubits, seed)

    @staticmethod
    def uninitialized_state(batch_size, n_qubits, precision='f64', space='default'):
        return _get_module(precision, space).StateVectorBatched.uninitialized_state(batch_size, n_qubits)

def Gate(gate):
    key = _key_for_gate(gate)
    if key is None:
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    return _modules[('default', key[1])].Gate(gate)

def ParamGate(gate):
    key = _key_for_gate(gate)
    if key is None:
        raise ValueError(f"Unsupported gate type: {type(gate)}")
    return _modules[('default', key[1])].ParamGate(gate)

def _make_gate_wrapper(name, is_param, is_space_specific):
    def caster(gate, space='default'):
        key = _key_for_gate(gate)
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

def merge_gate(gate1, gate2, prec='f64', space='default'):
    mod = _get_module(prec, space)
    if not hasattr(mod, 'merge_gate'):
        raise ValueError(f"merge_gate is not available in space={space} with precision={prec}")
    return mod.merge_gate(gate1, gate2)

class PauliOperator:
    def __new__(cls, coef=1.0, precision='f64'):
        return _get_module(precision, 'default').PauliOperator(coef=coef)

    @staticmethod
    def from_targets_and_pauli_ids(target_qubit_list, pauli_id_list, coef=1.0, precision='f64', space='default'):
        return _get_module(precision, space).PauliOperator(target_qubit_list, pauli_id_list, coef=coef)

    @staticmethod
    def from_pauli_string(pauli_string, coef=1.0, precision='f64', space='default'):
        return _get_module(precision, space).PauliOperator(pauli_string, coef=coef)
    
    @staticmethod
    def from_pauli_id_par_qubit(pauli_id_par_qubit, coef=1.0, precision='f64', space='default'):
        return _get_module(precision, space).PauliOperator(pauli_id_par_qubit, coef=coef)
    
    @staticmethod
    def from_XZ_mask(bit_flip_mask, phase_flip_mask, coef=1.0, precision='f64', space='default'):
        return _get_module(precision, space).PauliOperator(bit_flip_mask, phase_flip_mask, coef=coef)

class Operator:
    def __new__(cls, terms, precision='f64', space='default'):
        return _get_module(precision, space).Operator(terms)

class OperatorBatched:
    def __new__(cls, terms, precision='f64', space='default'):
        return _get_module(precision, space).OperatorBatched(terms)

class Circuit:
    def __new__(cls, precision='f64'):
        return _get_module(precision, 'default').Circuit()
