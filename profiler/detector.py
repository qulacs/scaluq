import scaluq as scaluqbase

class UnsupportedPrecisionError(Exception):
    pass

class UnsupportedSyclError(Exception):
    pass

class Detector:
    gpu_backends = {'cuda', 'hip', 'sycl'}

    def __init__(self, precision: str):
        if(scaluqbase.precision_available(precision) is False):
            raise UnsupportedPrecisionError()
        self.precision = precision

    def detect(self):
        default = scaluqbase.get_default_execution_space()
        if(default == 'host'): # cpu-only
            return ['default(host)', 'host_serial']

        elif (default in self.gpu_backends):
            if(default == 'sycl' and self.precision == 'f16'):
                raise UnsupportedSyclError()
            return [f"default({default})", 'host', 'host_serial']
