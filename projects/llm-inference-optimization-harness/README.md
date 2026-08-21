# Distributed LLM Serving & GPU Optimization System

A single-model ML systems project for serving and profiling `Qwen/Qwen3-8B` in BF16. It connects a reliability-focused FastAPI gateway, four data-parallel vLLM workers, a reproducible serving benchmark harness, and the repository's CUDA/NCCL FlashAttention-style microbenchmark into one traceable evidence path.

The implementation now includes the serving gateway, serving benchmark harness, hardened CUDA/NCCL microbenchmark, and validated one- and four-A10G AWS evidence sets. Every performance claim below links to the committed request-level or profiler artifact that supports it.

## System Boundary

The two execution paths share benchmark metadata and reporting, but they are intentionally separate:

1. **Serving path:** clients call the FastAPI gateway, which routes requests to vLLM workers. vLLM supplies model execution, continuous batching, paged KV-cache management, and token generation.
2. **Kernel path:** the custom C++/CUDA/NCCL code benchmarks FlashAttention-style streaming softmax and ring sequence parallelism independently of vLLM.

The custom CUDA kernel is not integrated into vLLM, and this project does not claim that continuous batching or vLLM's KV cache was implemented from scratch.

## Implemented Serving Features

- Streaming `/v1/chat/completions` and `/v1/completions` endpoints for one configured Qwen3-8B deployment.
- Four-worker data-parallel routing with stable-prefix affinity and a load-aware override.
- Bounded admission control with HTTP 429 rejection instead of an unbounded request queue.
- End-to-end deadlines, client-disconnect cleanup, health checks, and HTTP 503 availability responses.
- Failover only before the first generated token; a mid-stream failure is recorded and closed without replaying output from another worker.
- Prometheus request, route, queue, TTFT, E2E, and error metrics plus OpenTelemetry request spans.
- A deterministic fake backend for local correctness tests. Fake-backend timing is not accepted as performance evidence.

## Local Demo

Start the gateway and four deterministic fake workers with one command:

```bash
make serving-demo
```

In another terminal, send a streaming request:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Explain ring attention."}],"stream":true,"max_tokens":8}'
```

Inspect routing and health state:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

The same fake deployment can run in Docker:

```bash
docker compose -f compose.fake.yaml up --build
```

Run the complete local validation matrix in another terminal:

```bash
.venv/bin/python -m llm_serving_system.load_benchmark \
  --deployment local-fake-validation \
  --worker-count 4 \
  --backend-kind fake \
  --tokenizer whitespace
```

This sends the default 512-input/128-output workload at concurrency `1/8/16/32`, with one warm-up and five measured runs per concurrency. The fake backend intentionally caps generated test tokens; its report validates parsing and orchestration only and is always marked invalid for performance claims.

## Four-GPU vLLM Configuration

`compose.vllm.yaml` assigns one vLLM process to each GPU and points the gateway at all four workers:

```bash
docker compose -f compose.vllm.yaml up
```

Use `compose.vllm.single.yaml` for the single-GPU deployment. Before either measured run, set `VLLM_IMAGE` to a versioned tag or digest:

```bash
export VLLM_IMAGE='vllm/vllm-openai:<tested-version>'
docker compose -f compose.vllm.single.yaml up
```

The configuration serves only `Qwen/Qwen3-8B`, uses BF16, enables prefix caching, and caps model context at 4,096 tokens. The four-worker file is intended for a `g5.12xlarge` or equivalent four-GPU host, not this CPU-only laptop.

Gateway behavior is configured through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `SERVING_BACKEND_ENDPOINTS` | ports 8101-8104 | Comma-separated vLLM worker URLs |
| `SERVING_MODEL` | `Qwen/Qwen3-8B` | The only accepted model name |
| `SERVING_MAX_IN_FLIGHT` | `64` | Immediate admission limit |
| `SERVING_REQUEST_TIMEOUT_SECONDS` | `120` | End-to-end request deadline |
| `SERVING_FAILOVER_ATTEMPTS` | `2` | Maximum pre-token worker attempts |
| `SERVING_AFFINITY_LOAD_SLACK` | `2` | Allowed queue-depth difference before overriding affinity |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Optional OTLP/HTTP trace collector |

Run a real serving benchmark after the deployment is healthy:

```bash
python3 -m pip install -e '.[benchmark]'
llm-serving-benchmark \
  --deployment aws-a10g-1gpu \
  --worker-count 1 \
  --backend-kind vllm \
  --tokenizer huggingface \
  --hourly-cost-usd '<current-instance-hourly-price>' \
  --vllm-image "$VLLM_IMAGE" \
  --backend-metrics-url http://127.0.0.1:8101/metrics
```

For four workers, pass `--worker-count 4` and all four metrics URLs on ports `8101` through `8104`. The harness writes request-level CSV plus JSON and Markdown summaries under `serving-results/`. Real evidence is accepted only when the Qwen tokenizer, local GPU metadata, a pinned vLLM image, and fully successful measured runs are present.

## Tests

```bash
make test-all
```

The serving tests cover prefix routing, load override, admission rejection, pre-token failover, no mid-stream retry, deadlines, health state, SSE token detection, and single-model enforcement. The existing C++ tests continue to validate streaming attention and routed MoE against deterministic CPU references.

## Existing CPU Evidence

The portable C++17 harness compares:

- Materialized attention with numerically stable streaming attention.
- A ring sequence-parallel simulator with the materialized reference.
- Dense top-1 MoE routing with expert-sharded routing and scatter-back reconstruction.

Run it with:

```bash
make test
make bench
```

Artifacts are written to `results/benchmark_results.json`, `results/benchmark_results.csv`, and `results/benchmark_summary.md`. The latest checked-in evidence covers seven deterministic sweeps, reports zero maximum absolute drift on those tested CPU shapes, and estimates a 95.26% reduction in per-shard attention working memory at sequence length 1,024.

These CPU timings are not GPU speedup claims.

## CUDA/NCCL Path

The custom microbenchmark under `../../week_08/dist-flash-attn` now implements persistent buffers and NCCL communicators, warm-up plus repeated median/p95 timing, maximum per-device CUDA-event duration, PyTorch SDPA fixtures, exact/observed memory accounting, an overlap-versus-serialized K/V experiment, and Nsight collection scripts.

The CUDA path has been compiled and run on four NVIDIA A10Gs. The checked-in evidence includes PyTorch SDPA correctness on 1/2/4 GPUs, 1,024/2,048/4,096-token steady-state sweeps, allocator telemetry, overlap-versus-serialized K/V rotation, and an Nsight Systems trace.

The full limitation audit and phase status live in `docs/implementation_checklist.md`.

## AWS Safety

`scripts/aws_gpu_benchmark.py` provisions temporary GPU instances with explicit cost confirmation, a TTL shutdown, result download, automatic instance termination, temporary SSH keys, and an IP-restricted security group. The AWS account currently has a $50 monthly alert, but that alert is not a hard spending cap.

AWS approved the 48-vCPU G/VT quota used by the `g5.12xlarge` run. The one- and four-GPU runs validated successful and failed-launch cleanup paths: all EC2 instances terminated, attached volumes were deleted, and temporary keys and security groups were removed. The four-GPU instance ran for about 21 minutes 38 seconds before termination was requested; its prorated resource cost is estimated at `$2.05` before credits, while AWS billing data may post later.

## Measured Single-GPU Evidence

The committed evidence under `results/aws-a10g-1gpu-20260813/` was collected from BF16 Qwen3-8B on one NVIDIA A10G using the pinned vLLM image digest recorded in the report. The workload used exact 512-token inputs, exact 128-token outputs, one warm-up, and five measured runs per concurrency.

- `739.868` aggregate output tokens/s at concurrency 32, `25.37x` the request-at-a-time baseline.
- `352 ms` p95 TTFT, `41.5 ms` p95 TPOT, and `5.537 s` p95 E2E at concurrency 32.
- Zero failures or timeouts across 285 measured requests.
- An estimated `$0.378` per million output tokens at concurrency 32, based on measured workload time and the supplied `$1.006/hour` instance price; setup time is excluded.
- A `18.333 ms` median custom-attention forward at sequence length 4,096 and dimension 64.
- `95.26%` lower formula-based minimal state and `92.14%` lower explicit workspace than a full 4,096 x 4,096 FP32 score matrix on one GPU.

See the [serving report](results/aws-a10g-1gpu-20260813/serving/serving_benchmark.md), [CUDA report](results/aws-a10g-1gpu-20260813/cuda-attention/cuda_attention_benchmark.md), and [measurement notes](results/aws-a10g-1gpu-20260813/measurement_notes.md).

## Measured Four-GPU Evidence

The committed evidence under `results/aws-a10g-4gpu-20260821/` was collected on one `g5.12xlarge` with four NVIDIA A10Gs. It uses the same model, image digest, exact token lengths, warm-up, and measured-run counts as the single-GPU serving artifact.

- `786.408` aggregate output tokens/s, `243 ms` p95 TTFT, `39.2 ms` p95 TPOT, and `5.195 s` p95 E2E at concurrency 32.
- Zero failures or timeouts across 285 measured requests, with all four workers represented in request-level routing evidence.
- An estimated `$2.003` per million output tokens at concurrency 32 using the full `$5.672/hour` instance price; setup time is excluded.
- Compared with the controlled one-A10G concurrency-32 artifact, four workers increased throughput by `6.29%`, reduced p95 TTFT by `30.99%`, and reduced p95 E2E by `6.18%`. This is not linear serving throughput scaling and is reported as measured.
- The CUDA/NCCL sequence-4,096 sweep reduced median forward time from `18.348 ms` on one GPU to `11.092 ms` on two and `7.380 ms` on four, a `2.49x` four-GPU speedup.
- Overlapped K/V rotation reduced four-GPU median latency by `6.49%` versus the serialized mode (`7.380 ms` versus `7.892 ms`).
- The sequence-128 1/2/4-GPU correctness sweep had at most `2.235e-8` maximum absolute error against PyTorch SDPA.
- At sequence length 4,096, the formula-based minimal state is `98.8159%` smaller and the explicitly requested workspace is `98.0347%` smaller per GPU than a full 4,096 x 4,096 FP32 score matrix. These are memory-accounting results, not allocator-residency measurements.

The serving comparison also exposes a useful bottleneck: spreading one prefix-heavy workload over four independent vLLM replicas reduces batching and cache locality, so serving throughput rose only `6.29%` even though the standalone sequence-parallel kernel scaled `2.49x`. The gateway prioritizes affinity but overrides it under load, which produced a 75% affinity-route rate at concurrency 32.

See the [four-GPU serving report](results/aws-a10g-4gpu-20260821/serving/serving_benchmark.md), [CUDA/NCCL report](results/aws-a10g-4gpu-20260821/cuda-attention/cuda_attention_benchmark.md), [Nsight summary](results/aws-a10g-4gpu-20260821/nsight/attention-seq4096-4gpu.stats.txt), and [measurement notes](results/aws-a10g-4gpu-20260821/measurement_notes.md).

## Remaining Target

The configurable 10% CI performance-regression gate remains disabled until repeated controlled GPU runs establish a stable baseline. It is not claimed as an enforced result.
