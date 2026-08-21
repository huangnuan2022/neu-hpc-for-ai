# CUDA/NCCL Attention Benchmark

- Generated at: `2026-08-21T09:24:20.531395+00:00`
- Git SHA: `e056e29e326868f7a88405f6ae13b3a02f2d3bf7`
- GPU inventory: `GPU 0: NVIDIA A10G (UUID: GPU-48747d8a-80af-e446-bce7-e16a78b16e6b)
GPU 1: NVIDIA A10G (UUID: GPU-c237f8f3-056d-a445-cdc0-30847fa72d1d)
GPU 2: NVIDIA A10G (UUID: GPU-0e9f2ac8-d15f-a326-dfe7-824a07c5f3e9)
GPU 3: NVIDIA A10G (UUID: GPU-78f76ab7-02e2-0953-76d0-9c7729d9c1e3)`
- CUDA compiler: `nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Built on Tue_Jun_09_02:43:40_PM_PDT_2026
Cuda compilation tools, release 13.3, V13.3.73
Build cuda_13.3.r13.3/compiler.38244171_0`
- Warm-up / iterations: `5` / `20`

## Correctness Against PyTorch SDPA

| GPUs | max abs error | max rel error | reference |
| ---: | ---: | ---: | --- |
| 1 | 2.142e-08 | 8.378e-03 | pytorch_sdpa |
| 2 | 2.235e-08 | 8.378e-03 | pytorch_sdpa |
| 4 | 2.235e-08 | 7.081e-03 | pytorch_sdpa |

## Steady-State Performance

| seq | GPUs | overlap | median | p95 | speedup vs 1 GPU | minimal state reduction | explicit workspace reduction |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | True | 1.817 ms | 1.818 ms | 1.00x | 81.05% | 68.55% |
| 1024 | 2 | True | 0.920 ms | 0.928 ms | 1.98x | 90.53% | 84.28% |
| 1024 | 2 | False | 0.956 ms | 0.960 ms | 1.90x | 90.53% | 84.28% |
| 1024 | 4 | True | 0.945 ms | 0.955 ms | 1.92x | 95.26% | 92.14% |
| 1024 | 4 | False | 1.015 ms | 1.024 ms | 1.79x | 95.26% | 92.14% |
| 2048 | 1 | True | 5.466 ms | 5.471 ms | 1.00x | 90.53% | 84.28% |
| 2048 | 2 | True | 3.696 ms | 3.709 ms | 1.48x | 95.26% | 92.14% |
| 2048 | 2 | False | 3.841 ms | 3.850 ms | 1.42x | 95.26% | 92.14% |
| 2048 | 4 | True | 1.895 ms | 1.916 ms | 2.88x | 97.63% | 96.07% |
| 2048 | 4 | False | 2.026 ms | 2.038 ms | 2.70x | 97.63% | 96.07% |
| 4096 | 1 | True | 18.344 ms | 18.348 ms | 1.00x | 95.26% | 92.14% |
| 4096 | 2 | True | 11.092 ms | 11.153 ms | 1.65x | 97.63% | 96.07% |
| 4096 | 2 | False | 11.333 ms | 11.351 ms | 1.62x | 97.63% | 96.07% |
| 4096 | 4 | True | 7.380 ms | 7.402 ms | 2.49x | 98.82% | 98.03% |
| 4096 | 4 | False | 7.892 ms | 7.898 ms | 2.32x | 98.82% | 98.03% |

## Method

- CUDA/NCCL communicators, streams, events, and buffers are created before warm-up.
- Each sample uses CUDA events and reports the maximum elapsed duration across participating devices.
- K/V input staging is outside timed forward execution; state initialization, attention kernels, NCCL rotation, and output finalization are timed.
- The overlap comparison uses separate compute/communication streams and double-buffered K/V rotation.
- Minimal state is formula-based; explicit workspace is the exact sum of cudaMalloc requests; measured allocation deltas are retained in JSON.
