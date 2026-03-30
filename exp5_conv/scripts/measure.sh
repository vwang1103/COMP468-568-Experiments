#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$EXP_DIR/bin/dconv2d"

SPATIAL=(256 512 1024)
CHANNELS=(16 32 64)
FILTERS=(32 64 128)
IMPLS=(naive tiled)
K=3
STRIDE=1
PADDING=1

DATA_DIR="$EXP_DIR/data"
mkdir -p "$DATA_DIR"

LOG="$DATA_DIR/$(date +%Y%m%d_%H%M%S)_conv_sweep.csv"
echo "impl,height,width,cin,cout,k,stride,padding,time_ms,gflops" > "$LOG"

for h in "${SPATIAL[@]}"; do
  for cin in "${CHANNELS[@]}"; do
    for cout in "${FILTERS[@]}"; do
      for impl in "${IMPLS[@]}"; do
        echo "Running $impl H=${h} W=${h} Cin=${cin} Cout=${cout}"

        output=$("$BIN" \
          --height "$h" \
          --width "$h" \
          --channels "$cin" \
          --filters "$cout" \
          --ksize "$K" \
          --stride "$STRIDE" \
          --padding "$PADDING" \
          --impl "$impl" \
          --no-verify 2>&1)

        time_ms=$(echo "$output" | grep -oP 'Time\(ms\)=\K[0-9.]+' | head -n1 || true)
        gflops=$(echo "$output" | grep -oP 'GFLOP/s=\K[0-9.]+' | head -n1 || true)

        time_ms="${time_ms:-0}"
        gflops="${gflops:-0}"

        echo "${impl},${h},${h},${cin},${cout},${K},${STRIDE},${PADDING},${time_ms},${gflops}" >> "$LOG"
      done
    done
  done
done

echo "Results stored in $LOG"