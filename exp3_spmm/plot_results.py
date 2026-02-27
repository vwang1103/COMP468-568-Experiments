import sys
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py results/spmm_results.csv [out_dir]")
        sys.exit(1)

    path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "results"

    # impl -> list of (density, gflops, err)
    data = defaultdict(list) 

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            impl = r["impl"]
            density = float(r["density"])
            gflops = float(r["gflops"])
            err = float(r["max_err"])
            data[impl].append((density, gflops, err))

    # Plot: GFLOP/s vs density
    plt.figure()
    for impl, pts in data.items():
        pts.sort(key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=impl)

    plt.xscale("log")
    plt.xlabel("Density")
    plt.ylabel("GFLOP/s")
    plt.title("SpMM GFLOP/s vs Density")
    plt.grid(True)
    plt.legend()

    out_path = f"{out_dir}/gflops_vs_density.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()