#!/usr/bin/env python3
"""Plot cuda_bench CSV output without requiring a notebook."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: plot_results.py benchmark_results.csv", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        print(f"{path} contains no benchmark rows", file=sys.stderr)
        return 1

    labels = [f"{row['workload']}:{row['mode']}" for row in rows]
    timings = [float(row["milliseconds"]) for row in rows]

    _, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, timings)
    axis.set_ylabel("Milliseconds")
    axis.set_title("CUDA Benchmark Results")
    axis.tick_params(axis="x", labelrotation=30)
    plt.tight_layout()

    output = path.with_suffix(".png")
    plt.savefig(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
