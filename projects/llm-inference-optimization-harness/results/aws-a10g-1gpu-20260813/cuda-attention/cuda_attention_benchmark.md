# CUDA/NCCL Attention Benchmark

- Generated at: `2026-08-13T15:45:41.329035+00:00`
- Git SHA: `c5cbb271aa8cbe738ac74a69a74d53ee40ac7040`
- GPU inventory: `GPU 0: NVIDIA A10G (UUID: GPU-13053a70-a369-ca00-8cf0-3c9b4bc86715)`
- CUDA compiler: `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0`
- Warm-up / iterations: `5` / `20`

## Correctness Against PyTorch SDPA

| GPUs | max abs error | max rel error | reference |
| ---: | ---: | ---: | --- |
| 1 | 0.000e+00 | 8.378e-03 | pytorch_sdpa |

## Steady-State Performance

| seq | GPUs | overlap | median | p95 | speedup vs 1 GPU | minimal state reduction | explicit workspace reduction |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | True | 1.817 ms | 1.818 ms | 1.00x | 81.05% | 68.55% |
| 2048 | 1 | True | 5.469 ms | 5.511 ms | 1.00x | 90.53% | 84.28% |
| 4096 | 1 | True | 18.333 ms | 18.342 ms | 1.00x | 95.26% | 92.14% |

## Method

- CUDA/NCCL communicators, streams, events, and buffers are created before warm-up.
- Each sample uses CUDA events and reports the maximum elapsed duration across participating devices.
- K/V input staging is outside timed forward execution; state initialization, attention kernels, NCCL rotation, and output finalization are timed.
- The overlap comparison uses separate compute/communication streams and double-buffered K/V rotation.
- Minimal state is formula-based; explicit workspace is the exact sum of cudaMalloc requests; measured allocation deltas are retained in JSON.
