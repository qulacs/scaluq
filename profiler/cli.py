# python3 cli.py --precision f64 --omp-threads xx
import os
import sys
import subprocess
from pathlib import Path


import detector as detector_module
import analyzer as analyzer_module
import plot as plot_module

def run_benchmark(space: str, precision: str, omp_threads: int, min_nqubits: int, max_nqubits: int, result_dir: Path):
    env = os.environ.copy()
    env["SCALUQ_SPACE"] = space
    env["SCALUQ_PRECISION"] = precision
    env["PYTHONUNBUFFERED"] = "1"

    env["OMP_NUM_THREADS"] = str(omp_threads if(space == "host" or space == "default(host)") else 1)
    env["OMP_DYNAMIC"] = "FALSE"

    env["MIN_NQUBITS"] = f"{min_nqubits}"
    env["MAX_NQUBITS"] = f"{max_nqubits}"


    stem = f"{space}-{precision}"
    json_path = result_dir / f"{stem}.json"
    log_path = result_dir / f"{stem}.log"

    json_path.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)

    command = [
        sys.executable, "-m", "pytest",
        str(Path(__file__).with_name("benchmark.py")),
        f"--benchmark-json={json_path}",
        "--benchmark-min-rounds=5",
        "-q",
    ]

    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if result.returncode != 0:
        print(f"Benchmark failed: {space}", file=sys.stderr)
        print(f"See {log_path}", file=sys.stderr)
        return False

    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', default='f64')
    parser.add_argument('--omp-threads', type=int, default=None)
    parser.add_argument("--min-nqubits", type=int, default=4)
    parser.add_argument("--max-nqubits", type=int, default=21)
    args = parser.parse_args()

    if(args.omp_threads and args.omp_threads < 1):
        print("--omp-threads must be 1 or more")
        sys.exit(1)
    if(args.min_nqubits <= 1):
        print("--min-nqubits must be 2 or more")
        sys.exit(1)
    if(args.min_nqubits > args.max_nqubits):
        print("--min-nqubits must be smaller than --max-nqubits")
        sys.exit(1)

    try:
        detector = detector_module.Detector(args.precision)
        profile_spaces = detector.detect()
    except detector_module.UnsupportedPrecisionError as e:
        print("precision must be one of f64, f32, f16 or fb16")
        sys.exit(1)
    except detector_module.UnsupportedSyclError as e:
        print(f"SYCL(f16) is not supported")
        sys.exit(1)

    omp_threads = (
        args.omp_threads
        or int(os.environ.get("OMP_NUM_THREADS", 0))
        or os.cpu_count()
        or 1
    )

    result_dir = Path(__file__).parent / "result"
    result_dir.mkdir(parents=True, exist_ok=True)


    for space in profile_spaces:
        print(f"Benchmarking {space} ({args.precision})...")
        if(run_benchmark(space, args.precision, omp_threads, args.min_nqubits, args.max_nqubits, result_dir) is False):
            sys.exit(1)
    

    analyzer = analyzer_module.Analyzer(result_dir)
    analyzer.load(profile_spaces, args.precision)
    best_spaces = analyzer.get_best_spaces()

    for group in best_spaces:
        plot_module.plot_heatmap(best_spaces, profile_spaces, group, omp_threads, result_dir)
        plot_module.plot_graph(analyzer.data, group, omp_threads, result_dir)

