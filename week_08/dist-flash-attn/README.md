
# Distributed FlashAttention v2 (CUDA C, multi-GPU)

This is a minimal educational implementation of a **single-head, forward-only, distributed FlashAttention v2** in pure CUDA C with NCCL‑based multi‑GPU support.

The code focuses on clarity and a simple ring‑style sequence parallelism scheme rather than squeezing out every last drop of performance.

## What it does

- Implements a numerically stable FlashAttention‑style streaming softmax:
  - No explicit `seq_len x seq_len` attention matrix is materialized.
  - State is kept as running `(m, l, acc)` per query.
- Single‑GPU baseline kernel.
- Multi‑GPU forward pass using a ring:
  - Sequence dimension is partitioned across `N` GPUs.
  - Each GPU keeps only its local K/V shard.
  - Queries are also sharded; each GPU produces a disjoint slice of the output.
  - K/V shards are rotated in a ring using NCCL `send/recv`, so every query shard eventually attends over all key/value shards.
  - This keeps per‑GPU HBM roughly flat vs the single‑GPU case (aside from small buffers), while gaining speed from parallelism.

Limitations and simplifications:

- Forward pass only.
- Single attention head.
- Float32 only.
- Assumes `seq_len % num_gpus == 0`.
- Intended for a single node with 1–8 GPUs and NCCL.

## Build

Requirements:

- CUDA toolkit (11.x or newer).
- NCCL 2.x (typical on multi‑GPU servers).
- CMake 3.18+.

```bash
git clone <this-repo>
cd dist-flash-attn
mkdir build && cd build
cmake ..
make -j
```

This builds a single binary:

```bash
./dist_flash_attn [--seq L] [--dim D] [--gpus N]
```

Defaults:

- `L = 1024`
- `D = 64`
- `N = 2`

Example:

```bash
./dist_flash_attn --seq 2048 --dim 64 --gpus 4
```

## What the demo prints

On a machine with multiple GPUs you should see something like:

- Single‑GPU forward time in milliseconds.
- Multi‑GPU forward time.
- Approximate speedup factor.
- Maximum absolute difference between single‑GPU and multi‑GPU outputs.

Numerical differences should stay small (on the order of `1e-5`–`1e-6` for float32), since both paths use the same streaming‑softmax formulation.

## Files

- `include/flash_attn.h` – public interface for single‑ and multi‑GPU forward passes.
- `include/utils.h` – small helper macros for CUDA/NCCL error checking.
- `src/flash_attn.cu` – kernels and host code for FlashAttention:
  - Streaming softmax step kernel.
  - State init and finalization.
  - Single‑GPU and ring‑based multi‑GPU wrappers.
- `src/main.cu` – simple driver that:
  - Generates random Q/K/V.
  - Runs single‑GPU baseline on device 0.
  - Runs the multi‑GPU implementation.
  - Compares timing and numerical differences.

## How to turn this into a GitHub repo

This directory is already laid out like a small library/project. To publish it:

```bash
cd dist-flash-attn
git init
git add .
git commit -m "Initial distributed FlashAttention v2 implementation"
git remote add origin git@github.com:<your-username>/dist-flash-attn.git
git push -u origin main
```

From there you can share the GitHub URL or hook it into your own tooling as needed.
