#!/usr/bin/env python3
"""Create one Scaluq/Qulacs comparison image for each 4x4 SIMD path."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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
    figure, (timing_axis, speedup_axis) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    colors = {"f32": "tab:blue", "f64": "tab:orange"}
    for precision in ("f32", "f64"):
        rows = sorted(precision_rows.get(precision, []), key=lambda row: row["qubits"])
        if not rows:
            continue
        qubits = [row["qubits"] for row in rows]
        targets = rows[0]["targets"]
        color = colors[precision]
        timing_axis.plot(
            qubits,
            [row["scaluq"] for row in rows],
            marker="o",
            color=color,
            label=f"Scaluq {precision.upper()} targets={targets}",
        )
        timing_axis.plot(
            qubits,
            [row["qulacs"] for row in rows],
            marker="s",
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"Qulacs F64 targets={targets} ({precision.upper()} pair)",
        )
        speedup_axis.plot(
            qubits,
            [row["speedup"] for row in rows],
            marker="o",
            color=color,
            label=f"Qulacs / Scaluq {precision.upper()}",
        )

    timing_axis.set_yscale("log")
    timing_axis.set_ylabel("Median time per gate [us]")
    timing_axis.set_title(f"4x4 dense SIMD {path_name} path: Scaluq vs Qulacs")
    timing_axis.grid(True, which="both", alpha=0.3)
    timing_axis.legend()
    speedup_axis.axhline(1.0, color="black", linewidth=1)
    speedup_axis.set_xlabel("Number of qubits")
    speedup_axis.set_ylabel("Qulacs / Scaluq [x]")
    speedup_axis.grid(True, alpha=0.3)
    speedup_axis.legend()
    if path_name == "low":
        figure.text(
            0.5,
            0.01,
            "Note: Scaluq's 4x4 low path exists only for F32; Qulacs uses F64.",
            ha="center",
            fontsize=9,
        )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
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
