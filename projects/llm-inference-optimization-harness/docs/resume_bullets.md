# Resume Bullets

## Current Measured Version

**Distributed LLM Serving & GPU Optimization System**

**Technologies:** Python, FastAPI, vLLM, PyTorch, C++17, CUDA, NCCL, OpenTelemetry, Docker, AWS EC2, Prometheus, Nsight Systems, GitHub Actions

- Engineered a distributed LLM serving system around vLLM continuous batching with streaming APIs, prefix-affinity and load-aware routing, bounded backpressure, deadlines, cancellation, and health-aware pre-token failover across four data-parallel GPU workers.
- Deployed BF16 Qwen3-8B across 4x NVIDIA A10G workers, sustaining 786.4 output tokens/s at 32-request concurrency with 243 ms p95 TTFT and zero failures across 285 measured requests; reduced TTFT 31.0% versus a controlled one-GPU run at an estimated $2.00 per million output tokens.
- Optimized a FlashAttention-style CUDA/NCCL microbenchmark using streaming softmax, persistent buffers, and overlapped K/V rotation, reducing 4K-token median latency from 18.35 ms on one GPU to 7.38 ms on four GPUs (2.49x), cutting serialized latency 6.5%, and reducing per-GPU explicit workspace 98.03% versus a full FP32 score matrix.
- Built a reproducible performance harness across concurrency, sequence length, and 1/2/4-GPU sweeps; verified 2.24e-8 maximum absolute error against PyTorch SDPA, correlated Nsight profiles with Prometheus/vLLM telemetry, and automated cost-guarded AWS provisioning, artifact collection, TTL termination, and cleanup.
