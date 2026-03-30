#!/usr/bin/env python3
"""
Run spmm_baseline and spmm_opt, parse TIMING_CSV lines, and produce
a grouped bar chart comparing baseline vs warp-optimized kernels.

Usage:
    python3 plot_performance.py          # runs binaries and plots
    python3 plot_performance.py --help
"""

import subprocess
import sys
import os

# ---------- run binaries and collect timings ----------

def run_and_parse(binary):
    """Run a binary and return dict of {kernel_name: time_ms}."""
    result = subprocess.run(
        [os.path.join(".", binary)],
        capture_output=True, text=True, timeout=60
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR running {binary}:", result.stderr, file=sys.stderr)
        sys.exit(1)
    timings = {}
    for line in result.stdout.splitlines():
        if line.startswith("TIMING_CSV,"):
            parts = line.split(",")
            timings[parts[1]] = float(parts[2])
    return timings


timings = {}
for binary in ["spmm_baseline", "spmm_opt"]:
    timings.update(run_and_parse(binary))

print("\n=== Collected timings (ms) ===")
for k, v in timings.items():
    print(f"  {k:20s}  {v:.4f}")

# ---------- plot ----------

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

labels = ["SDDMM", "SpMM"]
baseline = [timings.get("baseline_sddmm", 0), timings.get("baseline_spmm", 0)]
optimized = [timings.get("warp_sddmm", 0), timings.get("warp_spmm", 0)]

x = np.arange(len(labels))
width = 0.30

fig, ax = plt.subplots(figsize=(6, 4))
bars1 = ax.bar(x - width / 2, baseline,  width, label="Baseline (naive)")
bars2 = ax.bar(x + width / 2, optimized, width, label="Warp-optimized")

ax.set_ylabel("Time (ms)")
ax.set_title("Kernel Performance: Baseline vs Warp-Optimized")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# annotate bars with values
for bar in bars1:
    ax.annotate(f"{bar.get_height():.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=8)

fig.tight_layout()
outpath = os.path.join("results", "performance.png")
os.makedirs("results", exist_ok=True)
fig.savefig(outpath, dpi=150)
print(f"\nPlot saved to {outpath}")

# ---------- also print speedup ----------
print("\n=== Speedup (baseline / warp) ===")
for lbl, b, o in zip(labels, baseline, optimized):
    if o > 0:
        print(f"  {lbl:8s}  {b/o:.2f}x")
    else:
        print(f"  {lbl:8s}  N/A (warp time is 0)")
