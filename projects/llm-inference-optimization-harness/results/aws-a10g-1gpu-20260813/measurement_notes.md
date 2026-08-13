# AWS A10G Measurement Notes

## Evidence Scope

- Commit: `c5cbb271aa8cbe738ac74a69a74d53ee40ac7040`
- Instance: `g5.xlarge`, one NVIDIA A10G, `us-east-1`
- Model: `Qwen/Qwen3-8B`, BF16
- Serving image: `vllm/vllm-openai:v0.26.0-cu129-ubuntu2404`; the inspected digest is retained under `environment/` and in the serving JSON.
- Serving artifact validity: `true`
- Measured serving requests: 285, all successful, all with 512 input and 128 output tokens.

The cost-per-million metric uses measured workload wall time and `$1.006/hour`; it excludes image/model download and setup. The two provisioning attempts ran for approximately 3.5 and 23.6 minutes, for an estimated combined EC2 and prorated gp3 cost below `$0.47` before credits. AWS billing data was not yet available when this note was written.

## CUDA Correctness Limitation

The PyTorch SDPA comparison reported `8.378e-3` maximum relative error. The original executable serialized floating-point values with six fixed decimal places, so the maximum absolute error was rounded to `0.000000`; this means it was below `5e-7`, not necessarily mathematically zero. Commit history after this artifact increases JSON precision for future runs.

The sequence-128 correctness case has more streaming state than the small full score matrix, so its memory-reduction percentages are negative and are not used as memory claims. Memory claims use the sequence-4,096 performance case.

## Profiling Interpretation

The Nsight trace includes process-level setup and teardown, so its CUDA API table contains `cudaMalloc` and `cudaFree`. The benchmark's reported steady-state samples use CUDA events after persistent workspace setup; allocation, NCCL initialization, and teardown are outside those event intervals.

## Pending Work

No multi-GPU speedup is claimed from this run. The 1/2/4-GPU NCCL matrix, overlap comparison, and 98.8% four-GPU memory claim require a `g5.12xlarge` run after the G/VT quota reaches 48 vCPUs.
