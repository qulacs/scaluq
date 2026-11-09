import json
from collections import defaultdict
from pathlib import Path

class Analyzer:
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir
        self.data = defaultdict(lambda: defaultdict(dict))


    def load(self, spaces: list[str], precision: str):
        for space in spaces:
            filepath = self.result_dir / f"{space}-{precision}.json"

            with filepath.open() as f:
                result = json.load(f)

            for item in result["benchmarks"]:
                group = item["group"]
                nqubits = int(item["params"]["nqubits"])
                niter = item["extra_info"]["niter"]

                gate_count = nqubits * 3
                elapsed = item["stats"]["median"]

                # time_per_gate (ms)
                time_per_gate = elapsed / (niter * gate_count) * 1000
                self.data[group][space][nqubits] = time_per_gate

        return self.data

    def get_best_spaces(self):
        best_spaces = {}

        for group, spaces in self.data.items():
            qubit_counts = set()

            for results in spaces.values():
                qubit_counts.update(results.keys())

            best_spaces[group] = {}
            best_spaces[group]["min_nqubits"] = min(qubit_counts)
            best_spaces[group]["max_nqubits"] = max(qubit_counts)
            best_spaces[group]["nqubits"] = {}

            for nqubits in sorted(qubit_counts):
                candidates = {
                    space: results[nqubits] for space, results in spaces.items() 
                    if nqubits in results
                }

                
                best_spaces[group]["nqubits"][nqubits] = {}
                best_spaces[group]["nqubits"][nqubits]['recommended'] = min(
                    candidates,
                    key=candidates.get,
                )

                best_spaces[group]["nqubits"][nqubits]['speedup'] = {
                    space: (candidates['host_serial'] / results[nqubits]) for space, results in spaces.items()
                    if nqubits in results
                }


        profile_path = self.result_dir / "profile.json"
        profile_path.unlink(missing_ok=True)
        with open(profile_path, mode="w") as f:
            json.dump(best_spaces, f, indent=4)

        return best_spaces
