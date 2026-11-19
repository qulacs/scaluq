import pytest
from unittest.mock import patch
from ..detector import Detector, UnsupportedSyclError, UnsupportedPrecisionError

def test_detect_host():
    with patch(
        "profiler.detector.scaluqbase.get_default_execution_space",
        return_value="host",
    ), patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=True,
    ):
        detector = Detector("f64")
        spaces = detector.detect()

    assert spaces == ["default(host)", "host_serial"]

def test_detect_cuda():
    with patch(
        "profiler.detector.scaluqbase.get_default_execution_space",
        return_value="cuda",
    ), patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=True,
    ):
        detector = Detector("f64")
        spaces = detector.detect()

    assert spaces == ["default(cuda)", "host", "host_serial"]

def test_detect_hip():
    with patch(
        "profiler.detector.scaluqbase.get_default_execution_space",
        return_value="hip",
    ), patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=True,
    ):
        detector = Detector("f64")
        spaces = detector.detect()

    assert spaces == ["default(hip)", "host", "host_serial"]

def test_detect_sycl():
    with patch(
        "profiler.detector.scaluqbase.get_default_execution_space",
        return_value="sycl",
    ), patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=True,
    ):
        detector = Detector("f64")
        spaces = detector.detect()

    assert spaces == ["default(sycl)", "host", "host_serial"]

def test_detect_sycl_error():
    with patch(
        "profiler.detector.scaluqbase.get_default_execution_space",
        return_value="sycl",
    ), patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=True,
    ):
        with pytest.raises(UnsupportedSyclError):
            detector = Detector("f16")
            spaces = detector.detect()

def test_detect_unsupported_error():
    with patch(
        "profiler.detector.scaluqbase.precision_available",
        return_value=False,
    ):
        with pytest.raises(UnsupportedPrecisionError):
            Detector("bf16")


