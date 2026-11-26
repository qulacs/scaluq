import pytest
from ..cli import run_benchmark
import scaluq as scaluqbase
import json

def test_run_benchmark(tmp_path):
    space = "host_serial"
    precisions = scaluqbase.get_available_precisions()
    precision = precisions[0]
    group = "Single-StateVector-Circuit"
    omp_threads = 16
    min_nqubits = 4
    max_nqubits = 4
    result_dir = tmp_path
    run_benchmark(group, space, precision, omp_threads, min_nqubits, max_nqubits, result_dir)

    stem = f"{group}-{space}-{precision}"
    json_file = result_dir / f"{stem}.json"
    log_file = result_dir / f"{stem}.log"

    assert json_file.exists()
    assert json_file.stat().st_size > 0
    with json_file.open() as f:
        result = json.load(f)

    assert len(result["benchmarks"]) == 1
    assert result["benchmarks"][0]["params"]["nqubits"] == min_nqubits
    assert result["benchmarks"][0]["extra_info"]["profile_space"] == space
    assert result["benchmarks"][0]["extra_info"]["precision"] == precision
    assert result["benchmarks"][0]["extra_info"]["omp_threads"] == "1"

    assert log_file.exists()
    assert log_file.stat().st_size > 0

