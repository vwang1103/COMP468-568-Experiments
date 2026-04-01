#!/usr/bin/env python3
"""
Run baseline vs optimized Two-Step GNN (SDDMM + SpMM) across multiple graph sizes,
collect averaged timing, and produce performance comparison plots.
"""

import subprocess, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ─── Graph generation parameters ───
GRAPH_CONFIGS = [
    # (num_nodes, avg_degree)
    (100,   10),
    (500,   15),
    (1000,  20),
    (2000,  20),
    (5000,  25),
    (8000,  25),
    (10000, 25),
]

EDGE_FILE = "graph_edges.txt"
DIR = os.path.dirname(os.path.abspath(__file__))

def generate_graph(M, avg_deg, seed=42):
    """Generate a random directed graph edge list."""
    import random
    random.seed(seed)
    edges = set()
    for u in range(M):
        n = random.randint(max(1, avg_deg // 2), avg_deg * 2)
        for _ in range(n):
            v = random.randint(0, M - 1)
            if v != u:
                edges.add((u, v))
    path = os.path.join(DIR, EDGE_FILE)
    with open(path, 'w') as f:
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")
    return len(edges)

def run_binary(name):
    """Run a binary and parse sddmm/spmm/total times from stdout."""
    result = subprocess.run(
        [os.path.join(DIR, name)],
        capture_output=True, text=True, cwd=DIR, timeout=300
    )
    out = result.stdout
    sddmm = spmm = total = None
    for line in out.splitlines():
        m = re.search(r'sddmm time:\s*([\d.]+)', line, re.IGNORECASE)
        if m: sddmm = float(m.group(1))
        m = re.search(r'spmm time:\s*([\d.]+)', line, re.IGNORECASE)
        if m: spmm = float(m.group(1))
        m = re.search(r'total time:\s*([\d.]+)', line, re.IGNORECASE)
        if m: total = float(m.group(1))
    if total is None and sddmm is not None and spmm is not None:
        total = sddmm + spmm
    return sddmm, spmm, total, out

def main():
    # Build first
    print("Building...")
    subprocess.run(["make", "clean"], cwd=DIR, capture_output=True)
    r = subprocess.run(["make"], cwd=DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print("Build failed:\n", r.stderr)
        sys.exit(1)
    print("Build OK\n")

    nodes_list = []
    nnz_list = []
    baseline_sddmm, baseline_spmm, baseline_total = [], [], []
    opt_sddmm, opt_spmm, opt_total = [], [], []

    for M, avg_deg in GRAPH_CONFIGS:
        nnz = generate_graph(M, avg_deg)
        print(f"=== M={M}, nnz={nnz}, avg_deg≈{avg_deg} ===")

        # Baseline
        b_sd, b_sp, b_tot, b_out = run_binary("spmm_baseline")
        print(f"  Baseline  — SDDMM: {b_sd:.4f} ms  SpMM: {b_sp:.4f} ms  Total: {b_tot:.4f} ms")

        # Optimized
        o_sd, o_sp, o_tot, o_out = run_binary("spmm_opt")
        print(f"  Optimized — SDDMM: {o_sd:.4f} ms  SpMM: {o_sp:.4f} ms  Total: {o_tot:.4f} ms")

        nodes_list.append(M)
        nnz_list.append(nnz)
        baseline_sddmm.append(b_sd); baseline_spmm.append(b_sp); baseline_total.append(b_tot)
        opt_sddmm.append(o_sd); opt_spmm.append(o_sp); opt_total.append(o_tot)

    # Convert to arrays
    nodes = np.array(nodes_list)
    b_sd = np.array(baseline_sddmm); b_sp = np.array(baseline_spmm); b_tot = np.array(baseline_total)
    o_sd = np.array(opt_sddmm);      o_sp = np.array(opt_spmm);      o_tot = np.array(opt_total)

    # ─── Save CSV ───
    csv_path = os.path.join(DIR, "results", "perf_comparison.csv")
    os.makedirs(os.path.join(DIR, "results"), exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write("nodes,nnz,baseline_sddmm_ms,baseline_spmm_ms,baseline_total_ms,"
                "opt_sddmm_ms,opt_spmm_ms,opt_total_ms\n")
        for i in range(len(nodes_list)):
            f.write(f"{nodes_list[i]},{nnz_list[i]},"
                    f"{baseline_sddmm[i]:.4f},{baseline_spmm[i]:.4f},{baseline_total[i]:.4f},"
                    f"{opt_sddmm[i]:.4f},{opt_spmm[i]:.4f},{opt_total[i]:.4f}\n")
    print(f"\nCSV saved to {csv_path}")

    # ─── Plot 1: Execution time comparison (SDDMM, SpMM, Total) ───
    x = np.arange(len(nodes))
    labels = [str(n) for n in nodes_list]
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, b_data, o_data, title in [
        (axes[0], b_sd, o_sd, "SDDMM"),
        (axes[1], b_sp, o_sp, "SpMM"),
        (axes[2], b_tot, o_tot, "Two-Step GNN Total"),
    ]:
        bars1 = ax.bar(x - width/2, b_data, width, label='Baseline', color='#d45f5f')
        bars2 = ax.bar(x + width/2, o_data, width, label='Optimized (Warp)', color='#5f8dd4')
        ax.set_xlabel('Number of Nodes')
        ax.set_ylabel('Time (ms)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(DIR, "results", "time_comparison.png")
    plt.savefig(path1, dpi=150)
    print(f"Plot saved: {path1}")
    plt.close()

    # ─── Plot 2: Speedup ───
    fig, ax = plt.subplots(figsize=(8, 5))
    speedup_sddmm = b_sd / o_sd
    speedup_spmm  = b_sp / o_sp
    speedup_total = b_tot / o_tot

    ax.plot(nodes, speedup_sddmm, 'o-', label='SDDMM Speedup', color='#e07b39', linewidth=2)
    ax.plot(nodes, speedup_spmm,  's-', label='SpMM Speedup',  color='#39a87c', linewidth=2)
    ax.plot(nodes, speedup_total, 'D-', label='Total Speedup',  color='#7b39e0', linewidth=2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1x (no speedup)')
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Speedup (Baseline / Optimized)')
    ax.set_title('Warp-Optimized Speedup over Baseline')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(DIR, "results", "speedup.png")
    plt.savefig(path2, dpi=150)
    print(f"Plot saved: {path2}")
    plt.close()

    # Print summary
    print("\n─── Speedup Summary ───")
    print(f"{'Nodes':>8} {'SDDMM':>8} {'SpMM':>8} {'Total':>8}")
    for i in range(len(nodes_list)):
        print(f"{nodes_list[i]:>8} {speedup_sddmm[i]:>7.2f}x {speedup_spmm[i]:>7.2f}x {speedup_total[i]:>7.2f}x")

if __name__ == "__main__":
    main()
