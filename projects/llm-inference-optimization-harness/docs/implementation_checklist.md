# Distributed LLM Serving & GPU Optimization System Checklist

This checklist separates implemented evidence from targets. Target values are not measured results.

## Phase 0: Baseline Audit

- [x] Preserved unrelated untracked course directories (`week9-deepseekmoe/`, `week11-flashdmoe/`).
- [x] Re-ran the existing C++17 smoke tests and seven deterministic CPU benchmark sweeps.
- [x] Confirmed the existing CPU evidence: zero reported absolute drift on tested shapes and a 95.26% estimated memory reduction at sequence length 1,024.
- [x] Audited `week_08/dist-flash-attn` and `week_07/deepseekmoe-labs/cuda_nccl`.
- [x] Confirmed the AWS runner has explicit cost confirmation, TTL shutdown, automatic result download, instance termination, temporary key deletion, and temporary security-group deletion.

### Known CUDA Benchmark Limitations

- The attention forward functions allocate/free workspace and initialize NCCL communicators and CUDA streams inside each call.
- The current executable records only one timed call and does not report warm-up, repeated samples, median, or p95.
- The multi-GPU timing uses CUDA events recorded on device 0 around host-side work spanning several devices; it is not a defensible distributed steady-state measurement.
- Correctness currently compares the multi-GPU result with the same custom single-GPU implementation, not PyTorch SDPA.
- K/V communication and attention compute use the same stream and are serialized; overlap has not yet been demonstrated.
- Existing memory reductions are formula-based estimates, not measured allocator telemetry.

## Phase 1: Serving System

- [x] Single-model guard for `Qwen/Qwen3-8B`.
- [x] Streaming `/v1/chat/completions` and `/v1/completions` proxy endpoints.
- [x] Configurable backend endpoints and four-worker data-parallel routing abstraction.
- [x] Bounded admission control with immediate HTTP 429 rejection.
- [x] End-to-end request deadline and stream cancellation cleanup.
- [x] Health checks and HTTP 503 when no backend is available.
- [x] Safe pre-token failover with no retry after token streaming begins.
- [x] Stable-prefix affinity with load-aware override and route-reason headers/metrics.
- [x] `/health`, Prometheus `/metrics`, and OpenTelemetry request spans.
- [x] Deterministic fake backend for local tests; fake results are never performance evidence.
- [x] Local four-worker demo and separate four-GPU vLLM Compose configuration.
- [ ] AWS Qwen3-8B BF16 validation (blocked on EC2 GPU quota).

## Phase 2: Serving Benchmark Harness

- [x] Async load generator with exact 512-token input IDs, 128-token output requests, and concurrency 1/8/16/32.
- [x] Warm-up plus at least five measured runs per configuration.
- [x] TTFT, TPOT, E2E, output tokens/s, requests/s, p50/p95/p99, error, timeout, queue, route, GPU, and cost metrics.
- [x] vLLM KV-cache, prefix-cache, running-request, and waiting-request metric sampling when exposed by workers.
- [x] JSON, CSV, and Markdown artifacts with configuration, git SHA, AMI, driver, CUDA, GPU, and pinned-image metadata.
- [x] Schema validation and request-level CSV evidence; fake-backend artifacts are explicitly invalid for performance claims.
- [ ] Measured single- and four-GPU Qwen3-8B artifacts (blocked on AWS quota).

## Phase 3: CUDA/NCCL Microbenchmark

- [x] Persistent/reused communicators, streams, events, and buffers with setup-free steady-state timing.
- [x] Warm-up, repeated samples, median/p95, and maximum per-device CUDA-event duration.
- [x] PyTorch SDPA fixture generation and full-output maximum absolute/relative error reporting.
- [x] Automated 1/2/4-GPU sweep definition for sequence length 1,024/2,048/4,096 and dimension 64.
- [x] Nsight Systems trace and stats collection script.
- [x] Separate-stream, double-buffered K/V rotation and serialized comparison mode.
- [x] Formula and CPU test for 98.8159% minimal-state and 98.0347% explicit double-buffer-workspace reductions at 4,096 tokens on four GPUs.
- [ ] Compile and correctness validation on NVIDIA hardware.
- [ ] Measured 1/2/4-GPU latency, scaling, allocation deltas, overlap benefit, and Nsight artifacts.

## Targets Pending Measurement

- 200+ output tokens/s at concurrency 32.
- Less than 2.0 seconds p95 TTFT.
- 2.0x throughput over sequential serving.
- Less than $10 per million output tokens.
- 2.0x four-GPU custom-attention speedup at sequence length 4,096.
- 10% CI performance-regression budget after a stable controlled baseline exists.
