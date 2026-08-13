# NEU HPC for AI

Course projects and polished follow-up systems work for high-performance AI inference.

## Featured Project

### Distributed LLM Serving & GPU Optimization System

Location: `projects/llm-inference-optimization-harness`

This project combines a real Qwen3-8B serving path with a separate CUDA/NCCL microbenchmark path. The serving gateway is implemented locally and ready to switch from deterministic fake backends to four data-parallel vLLM GPU workers. It includes:

- Streaming FastAPI inference endpoints with bounded admission control, deadlines, health-aware failover, and cancellation cleanup.
- Stable-prefix affinity routing with load-aware override across four worker endpoints.
- Prometheus metrics and OpenTelemetry request traces.
- FlashAttention-style streaming softmax correctness checks.
- Ring sequence-parallel attention simulation across logical shards.
- Top-1 MoE routing across expert shards.
- CPU-local benchmark artifacts for machines without CUDA.
- GitHub Actions gates for serving tests, build, smoke tests, and benchmark correctness.

vLLM supplies continuous batching, KV-cache management, and model execution. The custom CUDA/NCCL attention code is a separate microbenchmark and is not presented as integrated into vLLM.

Latest local benchmark summary:

- `7` correctness-gated sweeps across attention and MoE inference primitives.
- `0.000e+00` maximum observed attention and MoE output drift against reference implementations.
- `95.26%` estimated peak per-shard attention working-memory reduction versus materializing a `1024 x 1024` score matrix.

Run locally:

```bash
cd projects/llm-inference-optimization-harness
make serving-demo
```

Run all CPU-local tests with `make test-all`; run the existing benchmark artifacts with `make bench`.

Results are written to:

- `projects/llm-inference-optimization-harness/results/benchmark_summary.md`
- `projects/llm-inference-optimization-harness/results/benchmark_results.csv`
- `projects/llm-inference-optimization-harness/results/benchmark_results.json`

AWS GPU runbook with budget alerts, TTL cleanup, result download, and automatic EC2 termination:

- `projects/llm-inference-optimization-harness/docs/aws_gpu_runbook.md`

## Related CUDA/NCCL Implementations

- `week_08/dist-flash-attn`: distributed FlashAttention-style CUDA/NCCL implementation.
- `week_07/deepseekmoe-labs`: C, MPI, and CUDA/NCCL DeepSeekMoE-style routing lab.
