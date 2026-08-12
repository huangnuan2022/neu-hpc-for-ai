# LLM Inference Optimization Harness

Portable benchmark and correctness harness for LLM inference primitives that are common in modern AI infrastructure interviews and production systems work:

- FlashAttention-style streaming softmax.
- Ring-style sequence-parallel attention.
- Top-1 Mixture-of-Experts routing across logical expert shards.

The project is designed to turn the CUDA/NCCL course implementations in this repository into a resume-ready systems project with reproducible local evidence. The local harness is written in C++17 and has no third-party dependencies, so it runs on a CPU-only laptop. The CUDA/NCCL implementation path lives in:

- `../../week_08/dist-flash-attn`
- `../../week_07/deepseekmoe-labs`

## Why This Project Exists

Large SWE/MLE job postings increasingly emphasize ML systems, distributed inference, GPU programming, model-serving reliability, and benchmark-driven optimization. A notebook-only project usually does not prove those skills. This harness demonstrates the systems side: stable numerical algorithms, sharded execution, correctness gates, and benchmark artifacts that can be rerun.

## What It Verifies

### Attention

The attention harness compares three implementations on deterministic inputs:

1. A materialized softmax reference that builds the full `seq x seq` score matrix.
2. A streaming softmax path that keeps only running `(m, l, acc)` state.
3. A ring sequence-parallel simulator that partitions Q/K/V across logical shards and rotates K/V shards in the same order as the CUDA/NCCL implementation.

The correctness gate is the maximum absolute difference against the materialized reference.

### MoE

The MoE harness compares:

1. Dense top-1 routing reference.
2. Routed expert-sharded execution that buckets tokens by expert owner, processes local experts, and scatters outputs back.

The correctness gate is the maximum absolute difference against the dense reference, plus route-imbalance reporting.

## Run

```bash
make test
make bench
```

Benchmark artifacts are written under `results/`:

- `results/benchmark_results.json`
- `results/benchmark_results.csv`
- `results/benchmark_summary.md`

## Example Commands

```bash
build/inference_harness attention --seq 512 --dim 64 --shards 4 --iters 2
build/inference_harness moe --tokens 512 --dim 32 --hidden 64 --experts 8 --shards 4 --iters 2
```

Each command prints one JSON object so the harness can be used in CI, local profiling, or future agentic tuning loops.

## GPU Path

This laptop-safe harness validates the algorithmic contracts. For real GPU benchmarking, run the CUDA/NCCL code from the existing course projects:

```bash
cd ../../week_08/dist-flash-attn
mkdir -p build && cd build
cmake ..
make -j
./dist_flash_attn --seq 2048 --dim 64 --gpus 4
```

```bash
cd ../../week_07/deepseekmoe-labs
python3 tests/gen_cases.py --name case1 --N 2 --B 64 --d 32 --h 64 --E 8 --seed 1
make -C cuda_nccl
mpirun -np 2 ./cuda_nccl/moe_nccl --case cases/case1
```

## Resume-Safe Positioning

Use the generated `results/benchmark_summary.md` rather than guessed numbers. On CPU-only machines, claim correctness-gated benchmark coverage and memory-estimate reduction, not GPU speedup. On a multi-GPU box, add measured latency, throughput, and memory metrics from the CUDA/NCCL path.
