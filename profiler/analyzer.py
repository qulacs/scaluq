import json
from collections import defaultdict
from pathlib import Path

class Analyzer:
    def __init__(self, group: str, result_dir: Path, spaces: list[str], precision: str, threads: int):
        self.group = group
        self.result_dir = result_dir
        self.spaces = spaces
        self.precision = precision
        self.data = defaultdict(lambda: defaultdict(dict))
        self.threads = threads


    def load(self):
        for space in self.spaces:
            filepath = self.result_dir / f"{self.group}-{space}-{self.precision}.json"

            with filepath.open() as f:
                result = json.load(f)

            for item in result["benchmarks"]:
                group = item["group"]
                nqubits = int(item["params"]["nqubits"])
                niter = item["extra_info"]["niter"]

                if(group == "Single-StateVector-Circuit"):
                    # time_per_gate (ms)
                    gate_count = nqubits * 3
                    elapsed = item["stats"]["median"]
                    time = elapsed / (niter * gate_count) * 1000
                
                self.data[group][space][nqubits] = time

        return self.data

    def get_best_spaces(self):
        best_spaces = {}

        for group, spaces in self.data.items():
            qubit_counts = set()

            for results in spaces.values():
                qubit_counts.update(results.keys())

            best_spaces[group] = {}
            best_spaces[group][self.precision] = {}
            best_spaces[group][self.precision]["metadata"] = {}
            best_spaces[group][self.precision]["metadata"]["threads"] = self.threads
            best_spaces[group][self.precision]["min_nqubits"] = min(qubit_counts)
            best_spaces[group][self.precision]["max_nqubits"] = max(qubit_counts)
            best_spaces[group][self.precision]["nqubits"] = {}

            for nqubits in sorted(qubit_counts):
                candidates = {
                    space: results[nqubits] for space, results in spaces.items() 
                    if nqubits in results
                }

                
                best_spaces[group][self.precision]["nqubits"][nqubits] = {}
                best_spaces[group][self.precision]["nqubits"][nqubits]['recommended'] = min(
                    candidates,
                    key=candidates.get,
                )

                best_spaces[group][self.precision]["nqubits"][nqubits]['speedup'] = {
                    space: (candidates['host_serial'] / results[nqubits]) for space, results in spaces.items()
                    if nqubits in results
                }

            self.write_to_json(best_spaces, group, self.precision)

        return best_spaces

    def write_to_json(self, best_spaces, group, precision):

        profile_path = self.result_dir / "profile.json"
        if profile_path.exists():
            with open(profile_path, "r") as f:
                data = json.load(f)
        else:
            data = {}

        if(group not in data):
            data[group] = best_spaces[group]
        else:
            data[group][precision] = best_spaces[group][precision]

        with open(profile_path, mode="w") as f:
            json.dump(data, f, indent=4)
