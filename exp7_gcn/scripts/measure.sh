#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="./bin/dgcn"
VERIFY="python scripts/compare_with_dgl.py"
GRAPHS=("data/cora")
HIDDENS=(64 128 256)
IMPLS=(baseline fused)
LAYERS=2
TOL=1e-3

mkdir -p data
LOG="data/$(date +%Y%m%d_%H%M%S)_gcn_sweep.csv"

echo "graph,hidden,impl,time_ms,edges_per_s,spmm_gbps,dgl_time_ms,dgl_edges_per_ms,dgl_acc,cpp_acc,max_diff,mean_diff,match" > "$LOG"

extract_field() {
  local text="$1"
  local regex="$2"
  echo "$text" | sed -n "s|.*${regex}.*|\1|p" | head -n1
}

estimate_spmm_gbps() {
  local graph="$1"
  local hidden="$2"
  local time_ms="$3"

  python - "$graph" "$hidden" "$time_ms" <<'PY'
import os
import struct
import sys

graph = sys.argv[1]
hidden = int(sys.argv[2])
time_ms = float(sys.argv[3])

with open(graph + ".csr", "rb") as f:
    num_nodes = struct.unpack("i", f.read(4))[0]
    nnz = struct.unpack("i", f.read(4))[0]

feat_bytes = os.path.getsize(graph + ".feat")
feature_dim = feat_bytes // (num_nodes * 4)

# Rough effective bytes touched by the two sparse layers only.
# Layer 1: SpMM on [N x H]
# Layer 2: SpMM on [N x C], but we do not know C from header directly.
# Infer num_classes from labels.
label_bytes = os.path.getsize(graph + ".label")
num_labels = label_bytes // 4

# Read labels to infer num_classes
with open(graph + ".label", "rb") as f:
    labels = struct.unpack(f"{num_labels}i", f.read())
num_classes = max(labels) + 1

# For each SpMM, rough bytes:
# row_offsets + col_indices + values + dense input + dense output
# all int/float32 = 4 bytes
bytes_l1 = 4 * ((num_nodes + 1) + nnz + nnz + num_nodes * hidden + num_nodes * hidden)
bytes_l2 = 4 * ((num_nodes + 1) + nnz + nnz + num_nodes * num_classes + num_nodes * num_classes)

# This is a rough lower-bound traffic model for the sparse passes only.
total_bytes = bytes_l1 + bytes_l2

gbps = total_bytes / (time_ms / 1000.0) / 1e9
print(f"{gbps:.6f}")
PY
}

for graph in "${GRAPHS[@]}"; do
  for hidden in "${HIDDENS[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "============================================================"
      echo "Running impl=$impl graph=$graph hidden=$hidden"
      echo "============================================================"

      outputs_bin="data/tmp_$(basename "$graph")_${hidden}_${impl}.bin"

      cpp_output=$("$BIN" \
        --graph "$graph" \
        --hidden "$hidden" \
        --layers "$LAYERS" \
        --impl "$impl" \
        --dump "$outputs_bin" \
        --no-verify)

      echo "$cpp_output"

      summary_line=$(echo "$cpp_output" | grep '^Impl=')

      if [[ -z "$summary_line" ]]; then
        echo "ERROR: failed to find summary line for $graph hidden=$hidden impl=$impl" >&2
        exit 1
      fi

      time_ms=$(extract_field "$summary_line" 'Time(ms)=\([0-9.]*\)')
      edges_per_s=$(extract_field "$summary_line" 'Edges/s=\([0-9.]*\)')

      if [[ -z "$time_ms" || -z "$edges_per_s" ]]; then
        echo "ERROR: failed to parse time_ms or edges_per_s from summary line:" >&2
        echo "$summary_line" >&2
        exit 1
      fi

      spmm_gbps=$(estimate_spmm_gbps "$graph" "$hidden" "$time_ms")

      verify_graph="${graph}_dgl"
      set +e
      verify_output=$($VERIFY \
        --graph "$verify_graph" \
        --hidden "$hidden" \
        --layers "$LAYERS" \
        --outputs "$outputs_bin" \
        --tol "$TOL" 2>&1)
      verify_status=$?
      set -e

      echo "$verify_output"

      dgl_line=$(echo "$verify_output" | grep '^DGL')
      cpp_line=$(echo "$verify_output" | grep '^C++/cuSPARSE')

      if [[ -z "$dgl_line" || -z "$cpp_line" ]]; then
        echo "ERROR: failed to parse verifier output for $graph hidden=$hidden impl=$impl" >&2
        exit 1
      fi

      dgl_time_ms=$(extract_field "$dgl_line" 'time(ms)=\([0-9.]*\)')
      dgl_edges_per_ms=$(extract_field "$dgl_line" 'edges/ms=\([0-9.]*\)')
      dgl_acc=$(extract_field "$dgl_line" 'acc=\([0-9.]*\)')

      cpp_acc=$(extract_field "$cpp_line" 'acc=\([0-9.]*\)')
      max_diff=$(extract_field "$cpp_line" 'max_diff=\([0-9.]*\)')
      mean_diff=$(extract_field "$cpp_line" 'mean_diff=\([0-9.]*\)')

      if [[ $verify_status -eq 0 ]]; then
        match="yes"
      else
        match="no"
      fi

      echo "$graph,$hidden,$impl,$time_ms,$edges_per_s,$spmm_gbps,$dgl_time_ms,$dgl_edges_per_ms,$dgl_acc,$cpp_acc,$max_diff,$mean_diff,$match" >> "$LOG"
    done
  done
done

echo
echo "Results stored in $LOG"
