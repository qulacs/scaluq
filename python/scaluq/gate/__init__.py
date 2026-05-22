from .. import scaluq_core

_PRECISIONS = ['f64', 'f32', 'f16', 'bf16']
_SPACES = ['default', 'host', 'host_serial']
_available_precisions = [prec for prec in _PRECISIONS if scaluq_core.precision_available(prec)]

_modules: dict[tuple[str, str], object] = {}
for _space in _SPACES:
    _space_mod = getattr(scaluq_core, _space, None)
    if _space_mod is None:
        continue
    for _prec in _available_precisions:
        _prec_mod = getattr(_space_mod, _prec, None)
        if _prec_mod is None:
            continue
        _modules[(_space, _prec)] = _prec_mod.gate

def _get_module(precision, space):
    try:
        return _modules[(space, precision)]
    except KeyError:
        if space not in _SPACES:
            raise ValueError(f"Execution space {space} is not supported.")
        raise ValueError(f"Precision {precision} is not available.")

def _make_factory_wrapper(name):
    def factory(*args, precision='f64', **kwargs):
        mod = _get_module(precision, 'default')
        factory_func = getattr(mod, name, None)
        if factory_func is None:
            raise ValueError(f"{name} is not available in precision={precision}")
        return factory_func(*args, **kwargs)
    factory.__name__ = name
    factory.__qualname__ = name
    return factory

def _make_factory_wrapper_space_specific(name):
    def factory(*args, precision='f64', space='default', **kwargs):
        mod = _get_module(precision, space)
        factory_func = getattr(mod, name, None)
        if factory_func is None:
            raise ValueError(f"{name} is not available in precision={precision}, space={space}")
        return factory_func(*args, **kwargs)
    factory.__name__ = name
    factory.__qualname__ = name
    return factory

if len(_available_precisions) > 0:
    _one_prec = _available_precisions[0]
    _default_mod = _get_module(_one_prec, 'default')
    for _name in dir(_default_mod):
        if _name.startswith('_'):
            continue
        if hasattr(getattr(_default_mod, _name), '__call__'):
            if _name in ['SparseMatrix', 'DenseMatrix']:
                globals()[_name] = _make_factory_wrapper_space_specific(_name)
            else:
                globals()[_name] = _make_factory_wrapper(_name)
