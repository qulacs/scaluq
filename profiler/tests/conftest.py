import pytest
from pathlib import Path
from ..analyzer import Analyzer

@pytest.fixture
def analyzer(tmp_path) -> Analyzer:
    group = "Single-StateVector-Circuit"
    spaces = ['default(host)', 'host_serial']
    precision = "f64"
    threads = 16

    fixture_dir = Path(__file__).parent / "fixtures"
    for space in spaces:
        src = fixture_dir / f"{group}-{space}-{precision}.json"
        dst = tmp_path / src.name
        dst.write_bytes(src.read_bytes())


    instance = Analyzer(group, tmp_path, spaces, precision, threads)
    instance.load()

    return instance

