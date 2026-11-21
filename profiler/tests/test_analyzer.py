import pytest
import json
from pathlib import Path


def test_load(analyzer):
    data = analyzer.data
    group = analyzer.group

    assert group in data
    assert "host_serial" in data[group]
    assert "default(host)" in data[group]
    assert len(data[group]["host_serial"]) == 18
    assert set(data[group]["host_serial"].keys()) == set(range(4, 22))
    assert all(
        time > 0
        for time in data[group]["host_serial"].values()
    )
    assert len(data[group]["default(host)"]) == 18
    assert set(data[group]["default(host)"].keys()) == set(range(4, 22))
    assert all(
        time > 0
        for time in data[group]["default(host)"].values()
    )
    assert data[group]["host_serial"][4] == pytest.approx(0.0005132166677412897)
    assert data[group]["default(host)"][4] == pytest.approx(0.00847954999774932)
    

def test_get_best_spaces(analyzer):
    best_spaces = analyzer.get_best_spaces()

    group = analyzer.group
    precision = analyzer.precision
    threads = analyzer.threads

    test_data = best_spaces[group][precision]

    min_nqubits = 4
    max_nqubits = 21
    test_nqubits_list = list(range(min_nqubits, max_nqubits+1))
    expected_spaces = {4: "host_serial", 5: "host_serial", 6: "host_serial", 7: "host_serial", 8: "host_serial", 9: "host_serial", 10: "host_serial", 11: "host_serial", 12: "host_serial", 13: "host_serial",
                       14: "default(host)", 15: "default(host)", 16: "default(host)", 17: "default(host)", 18: "default(host)", 19: "default(host)", 20: "default(host)", 21: "default(host)"}
    for nqubits in test_nqubits_list:
        assert test_data["nqubits"][nqubits]["recommended"] == expected_spaces[nqubits]

    assert test_data["metadata"]["threads"] == threads
    assert test_data["min_nqubits"] == min_nqubits
    assert test_data["max_nqubits"] == max_nqubits
    assert test_data["nqubits"][4]["speedup"]["host_serial"] == pytest.approx(1.0)
    assert test_data["nqubits"][4]["speedup"]["default(host)"] == pytest.approx(0.06052404524739048)
    


def test_write_to_json(tmp_path, analyzer):
    best_spaces = analyzer.get_best_spaces()

    analyzer.write_to_json(best_spaces)

    target_file = tmp_path / "profile.json"
    assert target_file.exists()

    fixture_dir = Path(__file__).parent / "fixtures"
    expected_file = fixture_dir / "profile.json"

    with open(target_file, "r") as target_f, open(expected_file, "r") as expected_f:
        data = json.load(target_f)
        expected_data = json.load(expected_f)

    assert data == expected_data

