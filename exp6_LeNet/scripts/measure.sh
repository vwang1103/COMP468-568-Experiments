#!/usr/bin/env bash
set -u

BIN="../bin/dlenet"
BATCHES=(32 64 128)
ALGOS=(implicit_gemm implicit_precomp fft)
IMPLS=(baseline fused)

mkdir -p ../data
STAMP=$(date +%Y%m%d_%H%M%S)

FULL_LOG="../data/${STAMP}_lenet_full.csv"
CONV_LOG="../data/${STAMP}_lenet_conv_ref.csv"

echo "impl,batch,algo,verify,time_ms,full_gflops,workspace_bytes" > "$FULL_LOG"
echo "batch,cudnn_implicit_gemm_ref_gflops" > "$CONV_LOG"

for batch in "${BATCHES[@]}"; do
  conv_written=0

  for algo in "${ALGOS[@]}"; do
    for impl in "${IMPLS[@]}"; do

      echo "Running impl=$impl batch=$batch algo=$algo"

      output=$("$BIN" --batch "$batch" --algo "$algo" --impl "$impl" 2>&1)
      status=$?

      if [[ $status -ne 0 ]]; then
        echo "  -> unsupported or failed, skipping"
        continue
      fi

      verify_line=$(echo "$output" | grep "Verification" | tail -n 1)
      summary_line=$(echo "$output" | grep "Impl=" | tail -n 1)
      conv_line=$(echo "$output" | grep "conv-only" | head -n 1)

      if [[ -z "$verify_line" || -z "$summary_line" ]]; then
        echo "  -> missing output, skipping"
        continue
      fi

      if ! echo "$verify_line" | grep -q "PASS"; then
        echo "  -> verification failed, skipping"
        continue
      fi

      time_ms=$(echo "$summary_line" | sed -n 's/.*Time(ms)=\([0-9.]*\).*/\1/p')
      full_gflops=$(echo "$summary_line" | sed -n 's/.*GFLOP\/s=\([0-9.]*\).*/\1/p')
      workspace_bytes=$(echo "$summary_line" | sed -n 's/.*Workspace(bytes)=\([0-9]*\).*/\1/p')

      if [[ -z "$time_ms" || -z "$full_gflops" || -z "$workspace_bytes" ]]; then
        echo "  -> parse failed, skipping"
        continue
      fi

      echo "  -> verify=PASS time=${time_ms} ms, full_gflops=${full_gflops}, workspace=${workspace_bytes} bytes"
      echo "$impl,$batch,$algo,PASS,$time_ms,$full_gflops,$workspace_bytes" >> "$FULL_LOG"

      # Write the conv-only implicit_gemm reference once per batch.
      if [[ $conv_written -eq 0 && -n "$conv_line" ]]; then
        conv_gflops=$(echo "$conv_line" | sed -n 's/.*GFLOP\/s=\([0-9.]*\).*/\1/p')
        if [[ -n "$conv_gflops" ]]; then
          echo "$batch,$conv_gflops" >> "$CONV_LOG"
          conv_written=1
          echo "  -> recorded conv-only reference for batch=$batch: ${conv_gflops} GFLOP/s"
        fi
      fi

    done
  done
done

echo "Full-network results stored in $FULL_LOG"
echo "Conv-only reference results stored in $CONV_LOG"