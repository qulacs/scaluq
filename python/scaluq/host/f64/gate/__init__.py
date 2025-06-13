import scaluq
if scaluq.get_default_execution_space() == 'cuda':
    from ....scaluq_core.host.f64.gate import *
else:
    from ....scaluq_core.default.f64.gate import *
