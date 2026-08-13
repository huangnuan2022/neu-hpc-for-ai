# Resume Bullets

## Current Measured Version

**Distributed LLM Serving & GPU Optimization System**

**Technologies:** Python, FastAPI, vLLM, PyTorch, C++17, CUDA, NCCL, OpenTelemetry, Docker, AWS EC2, Prometheus, Nsight Systems, GitHub Actions

- Engineered a distributed LLM serving system on vLLM with streaming inference APIs, continuous batching, prefix-affinity routing, bounded backpressure, deadlines, cancellation, and health-aware pre-token failover across configurable data-parallel GPU workers.
- Benchmarked BF16 Qwen3-8B on one NVIDIA A10G at 739.9 aggregate output tokens/s across 32 concurrent requests, with 352 ms p95 TTFT, 25.37x throughput over request-at-a-time serving, zero failures across 285 measured requests, and an estimated $0.38 per million output tokens.
- Implemented a FlashAttention-style CUDA/NCCL microbenchmark with numerically stable streaming softmax, persistent buffers, CUDA-event timing, and double-buffered K/V rotation; achieved an 18.33 ms median at 4,096 tokens with 95.26% lower minimal state and 92.14% lower explicit workspace than a full FP32 score matrix on one GPU.
- Built a reproducible serving and kernel performance harness across concurrency and sequence-length sweeps; validated against PyTorch SDPA, correlated Nsight profiles with Prometheus/vLLM telemetry, added a controlled regression checker, and automated cost-guarded AWS provisioning, artifact collection, TTL termination, and cleanup.

## Four-GPU Draft, Not Yet Usable

Do not place four-GPU throughput, speedup, failure-rate, or 98.8% memory numbers on a submitted resume until the 48-vCPU quota is approved and the committed 1/2/4-GPU artifacts validate them. Replace this section with measured values after that run.
