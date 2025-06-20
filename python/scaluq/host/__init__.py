import scaluq as _scaluq
if _scaluq.get_default_execution_space() == 'cuda':
    from ..scaluq_core.host import *
else:
    from ..scaluq_core.default import *
if _scaluq.precision_available('f16'):
    from . import f16
if _scaluq.precision_available('f32'):
    from . import f32
if _scaluq.precision_available('f64'):
    from . import f64
if _scaluq.precision_available('bf16'):
    from . import bf16
