from ..scaluq_core.host_serial import *
import scaluq as _scaluq
if _scaluq.precision_available('f16'):
    from . import f16
if _scaluq.precision_available('f32'):
    from . import f32
if _scaluq.precision_available('f64'):
    from . import f64
if _scaluq.precision_available('bf16'):
    from . import bf16
