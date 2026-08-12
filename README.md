# NEU HPC for AI

Course projects and polished follow-up systems work for high-performance AI inference.

## Featured Project

### LLM Inference Optimization Harness

Location: `projects/llm-inference-optimization-harness`

This project turns the course CUDA/NCCL attention and MoE implementations into a reproducible benchmark and correctness harness for modern LLM inference systems. It includes:

- FlashAttention-style streaming softmax correctness checks.
- Ring sequence-parallel attention simulation across logical shards.
- Top-1 MoE routing across expert shards.
- CPU-local benchmark artifacts for machines without CUDA.
- GitHub Actions regression gate for build, smoke tests, and benchmark correctness.

Latest local benchmark summary:

- `7` correctness-gated sweeps across attention and MoE inference primitives.
- `0.000e+00` maximum observed attention and MoE output drift against reference implementations.
- `95.26%` estimated peak per-shard attention working-memory reduction versus materializing a `1024 x 1024` score matrix.

Run locally:

```bash
cd projects/llm-inference-optimization-harness
make test
make bench
```

Results are written to:

- `projects/llm-inference-optimization-harness/results/benchmark_summary.md`
- `projects/llm-inference-optimization-harness/results/benchmark_results.csv`
- `projects/llm-inference-optimization-harness/results/benchmark_results.json`

## Related CUDA/NCCL Implementations

- `week_08/dist-flash-attn`: distributed FlashAttention-style CUDA/NCCL implementation.
- `week_07/deepseekmoe-labs`: C, MPI, and CUDA/NCCL DeepSeekMoE-style routing lab.
