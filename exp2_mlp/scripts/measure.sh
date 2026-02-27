#!/usr/bin/env bash
set -euo pipefail

BIN="../bin/dmlp"
LAYERS=("512,512,512" "1024,2048,1024" "2048,2048,2048")
BATCHES=(64 128 256 512)
IMPLS=(baseline activation_fused)
ACTIVATION="gelu"

mkdir -p ../data
LOG="../data/$(date +%Y%m%d_%H%M%S)_mlp_sweep.csv"
echo "impl,layers,batch,activation,time_ms,gflops,max_abs_diff" > "$LOG"

median_of_5() { sort -n | awk 'NR==3{print; exit}'; }

for layers in "${LAYERS[@]}"; do
  layers_fmt=$(echo "$layers" | sed 's/,/, /g')

  for batch in "${BATCHES[@]}"; do
    for impl in "${IMPLS[@]}"; do
      echo "Performance: $impl layers=$layers batch=$batch act=$ACTIVATION (median of 5, --no-verify)"

      # ---- Performance sweep: 5 runs, verify OFF ----
      times=()
      gflops_list=()
      for run in {1..5}; do
        out=$("$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl" --no-verify)
        echo "$out"
        times+=("$(echo "$out" | sed -n 's/.*Time(ms)=\([0-9.]\+\).*/\1/p' | tail -n 1)")
        gflops_list+=("$(echo "$out" | sed -n 's/.*GFLOP\/s=\([0-9.]\+\).*/\1/p' | tail -n 1)")
      done
      time_ms=$(printf "%s\n" "${times[@]}" | median_of_5)
      gflops=$(printf "%s\n" "${gflops_list[@]}" | median_of_5)

      # ---- Correctness pass: 1 run, verify ON, capture max_abs_diff ----
      # (This is done AFTER perf so it doesn't perturb perf timing.)
      echo "Correctness: $impl layers=$layers batch=$batch act=$ACTIVATION (single run, verify ON)"
      out_verify=$("$BIN" --layers "$layers" --batch "$batch" --activation "$ACTIVATION" --impl "$impl")
      echo "$out_verify"
      max_abs_diff=$(
        echo "$out_verify" \
          | sed -n 's/.*max_abs_diff=\([0-9.eE+-]\+\).*/\1/p' \
          | head -n 1
      )

      echo "$impl,\"$layers_fmt\",$batch,$ACTIVATION,$time_ms,$gflops,$max_abs_diff" >> "$LOG"
    done
  done
done

echo "Results stored in $LOG"