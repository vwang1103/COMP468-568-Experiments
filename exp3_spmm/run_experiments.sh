#!/bin/bash
set -euo pipefail

make

OUTDIR="results"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/spmm_results.csv"

echo "impl,M,K,N,density,nnz,avg_ms,gflops,max_err" > "$CSV"

M=512
K=512
N=64

DENSITIES=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 1.0)

echo "Running SpMM experiment..."
echo "M=$M K=$K N=$N"
echo "Writing to $CSV"

run_and_append() {
  local impl="$1"
  local exe="$2"
  local d="$3"

  # Capture the program output, then extract numbers and write one CSV row
  local out
  out="$("$exe" "$M" "$K" "$N" "$d")"

  # Extract values from the known output format
  local nnz avg_ms gflops max_err
  nnz="$(echo "$out"    | awk -F'=' '/^nnz/ {gsub(/ /,"",$2); print $2; exit}')"
  avg_ms="$(echo "$out" | awk -F'=' '/^Avg kernel time/ {gsub(/ /,"",$2); print $2; exit}')"
  gflops="$(echo "$out" | awk -F'=' '/^GFLOP\/s/ {gsub(/ /,"",$2); print $2; exit}')"
  max_err="$(echo "$out"| awk -F'=' '/^Max error/ {gsub(/ /,"",$2); print $2; exit}')"

  echo "${impl},${M},${K},${N},${d},${nnz},${avg_ms},${gflops},${max_err}" >> "$CSV"
}

for d in "${DENSITIES[@]}"; do
  echo "density=$d"
  run_and_append "baseline" "./spmm_baseline" "$d"
  run_and_append "opt"      "./spmm_opt"      "$d"
done

echo "Done sweep."
echo "CSV saved to: $CSV"

# Run plotting automatically
echo "Running: python3 plot_results.py $CSV $OUTDIR"
python3 plot_results.py "$CSV" "$OUTDIR"

echo "If plotting succeeded, check:"
echo "  $OUTDIR/gflops_vs_density.png"