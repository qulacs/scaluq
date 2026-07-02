import scaluq as _scaluq
if _scaluq.get_default_execution_space() == 'cuda':
    from ....scaluq_core.host.f32.gate import *
elif _scaluq.get_default_execution_space() == 'sycl':
    from ....scaluq_core.host.f32.gate import *
else:
    from ....scaluq_core.default.f32.gate import *
