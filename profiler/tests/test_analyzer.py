import pytest
import json
from pathlib import Path
from ..analyzer import Analyzer


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

def test_write_to_json_update(tmp_path, analyzer):
    best_spaces = analyzer.get_best_spaces()
    analyzer.write_to_json(best_spaces)

    group = "Single-StateVector-Circuit"
    spaces = ['default(host)', 'host_serial']
    precision = "f32"
    threads = 16
    nqubits = 4
    best_spaces_update = {}
    best_spaces_update[group] = {}
    best_spaces_update[group][precision] = {}
    best_spaces_update[group][precision]["metadata"] = {}
    best_spaces_update[group][precision]["metadata"]["threads"] = threads
    best_spaces_update[group][precision]["min_nqubits"] = nqubits
    best_spaces_update[group][precision]["max_nqubits"] = nqubits
    best_spaces_update[group][precision]["nqubits"] = {}
    best_spaces_update[group][precision]["nqubits"][str(nqubits)] = {}
    best_spaces_update[group][precision]["nqubits"][str(nqubits)]["recommended"] = "host_serial"
    best_spaces_update[group][precision]["nqubits"][str(nqubits)]["speedup"] = {"default(host)": 0.01, "host_serial": 1.0,}

    analyzer_update = Analyzer(group, tmp_path, spaces, precision, threads)
    analyzer_update.write_to_json(best_spaces_update)

    target_file = tmp_path / "profile.json"
    assert target_file.exists()

    fixture_dir = Path(__file__).parent / "fixtures"
    before_file = fixture_dir / "profile.json"

    with open(target_file, "r") as target_f, open(before_file, "r") as before_f:
        data = json.load(target_f)
        before_data = json.load(before_f)

    assert group in data
    assert "f64" in data[group]
    assert "f32" in data[group]
    assert data[group]["f64"] == before_data[group]["f64"]
    assert data[group]["f32"] == best_spaces_update[group]["f32"]
