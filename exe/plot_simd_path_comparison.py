#!/usr/bin/env python3
"""Create one Scaluq/Qulacs comparison image for each 4x4 SIMD path."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def read_rows(path):
    grouped = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[row["path"]][row["precision"]].append(
                {
                    "qubits": int(row["qubits"]),
                    "scaluq": float(row["scaluq_median_us"]),
                    "qulacs": float(row["qulacs_median_us"]),
                    "speedup": float(row["speedup"]),
                    "targets": row["targets"].replace(";", ","),
                }
            )
    return grouped


def plot_path(path_name, precision_rows, output):
    figure, timing_axis = plt.subplots(figsize=(9, 6))
    precision_linestyles = {"f32": "--", "f64": "-"}
    for precision in ("f32", "f64"):
        rows = sorted(precision_rows.get(precision, []), key=lambda row: row["qubits"])
        if not rows:
            continue
        qubits = [row["qubits"] for row in rows]
        targets = rows[0]["targets"]
        linestyle = precision_linestyles[precision]
        timing_axis.plot(
            qubits,
            [row["scaluq"] / 1000.0 for row in rows],
            marker="P",
            linestyle=linestyle,
            color="tab:red",
            label=f"Scaluq {precision.upper()} targets={targets}",
        )
    qulacs_rows = precision_rows.get("f64") or precision_rows.get("f32")
    if qulacs_rows:
        qulacs_rows = sorted(qulacs_rows, key=lambda row: row["qubits"])
        timing_axis.plot(
            [row["qubits"] for row in qulacs_rows],
            [row["qulacs"] / 1000.0 for row in qulacs_rows],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label=f"Qulacs F64 targets={qulacs_rows[0]['targets']}",
        )

    timing_axis.set_yscale("log")
    timing_axis.set_xlabel("Number of qubits")
    timing_axis.set_ylabel("Execution time per gate [ms]")
    timing_axis.set_title(f"4x4 dense SIMD {path_name} path: Scaluq vs Qulacs")
    timing_axis.grid(True, which="both", alpha=0.3)
    timing_axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--output-dir", default="benchmark-results")
    arguments = parser.parse_args()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = read_rows(arguments.csv)
    for path_name in ("low", "middle", "high"):
        if path_name not in grouped:
            raise SystemExit(f"CSV contains no {path_name} rows")
        output = output_dir / f"comparison-{path_name}.png"
        plot_path(path_name, grouped[path_name], output)
        print(output)


if __name__ == "__main__":
    main()
