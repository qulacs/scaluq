import scaluq as _scaluq
if _scaluq.get_default_execution_space() == 'cuda':
    from ...scaluq_core.host.f16 import *
else:
    from ...scaluq_core.default.f16 import *
from . import gate
