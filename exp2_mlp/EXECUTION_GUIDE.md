# exp2_mlp — cuBLAS-Powered MLP Experiment

This lab benchmarks two MLP forward-pass schedules across multiple layer shapes and batch sizes. It runs a short performance sweep (median of 5 runs with verification disabled), then runs one correctness pass (verification enabled) to record numerical error. Results are written to a timestamped CSV under `data/`.

The tested implementations:

- `baseline` — per-layer `GEMM → bias-add → activation`
- `activation_fused` — per-layer `GEMM → fused(bias+activation)`

## 1) Repo layout

```
exp2_mlp/
	bin/         # built binary goes here (bin/dmlp)
	data/        # CSV outputs written here
	scripts/
		measure.sh  # sweep script
	src/         # CUDA source files
	Makefile     # build rules
```

## 2) Build the binary

From the `exp2_mlp/` directory:

```bash
make clean && make
```

## 3) Run a quick sanity check

From `exp2_mlp/`, run each implementation once:

```bash
./bin/dmlp --layers 1024,2048,1024 --batch 128 --activation gelu --impl baseline
./bin/dmlp --layers 1024,2048,1024 --batch 128 --activation gelu --impl activation_fused
```

Expected output includes lines like:

```
Verify: max_abs_diff=... max_rel_diff=...
Time(ms)=... GFLOP/s=...
```

## 4) Run the full sweep (`measure.sh`)

Make the script executable and run it from `exp2_mlp/`:

```bash
chmod +x scripts/measure.sh
./scripts/measure.sh
```

What the script does:

- Iterates layer lists:
  - `512,512,512`
  - `1024,2048,1024`
  - `2048,2048,2048`
- Iterates batch sizes: `64`, `128`, `256`, `512`
- Iterates implementations: `baseline`, `activation_fused`
- Uses activation: `gelu`
- For performance: runs 5 times with `--no-verify` and logs the median `time_ms` and median `gflops`
- For correctness: runs once with verification enabled and logs `max_abs_diff`
- Writes results to: `exp2_mlp/data/YYYYMMDD_HHMMSS_mlp_sweep.csv`

## 6) Output CSV

Results are saved under `exp2_mlp/data/`.

Example header:

```
impl,layers,batch,activation,time_ms,gflops,max_abs_diff
```

Example rows:

```
baseline,"1024, 2048, 1024",128,gelu,1.23,5432.10,1.2e-06
activation_fused,"1024, 2048, 1024",128,gelu,1.15,5810.77,1.2e-06
```
