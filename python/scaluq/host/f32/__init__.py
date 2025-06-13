import scaluq
if scaluq.get_default_execution_space() == 'cuda':
    from ...scaluq_core.host.f32 import *
else:
    from ...scaluq_core.default.f32 import *
from . import gate
