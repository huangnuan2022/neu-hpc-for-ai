# Serving Benchmark Report

- Evidence status: **Measured vLLM performance evidence.**
- Generated at: `2026-08-13T16:01:21.840765+00:00`
- Deployment: `aws-a10g-1gpu`
- Model: `Qwen/Qwen3-8B`
- Backend: `vllm` with `1` worker(s)
- Workload: `512` input tokens, `128` requested output tokens
- Concurrency: `[1, 8, 16, 32]`
- Runs: `1` warm-up + `5` measured per concurrency
- Git SHA: `c5cbb271aa8cbe738ac74a69a74d53ee40ac7040` (dirty: `False`)
- AWS: `{'instance_id': 'i-003cd6119babfdafc', 'instance_type': 'g5.xlarge', 'ami_id': 'ami-07bcf82131b289395', 'region': 'us-east-1'}`
- GPUs: `[{'index': 0, 'name': 'NVIDIA A10G', 'driver_version': '595.71.05', 'memory_total_mib': 23028.0}]`
- CUDA toolkit: `13.3`
- CUDA driver API: `13.2`
- vLLM image: `{'requested': 'vllm/vllm-openai:v0.26.0-cu129-ubuntu2404', 'image_id': 'sha256:f21f5e1987142d4a7c77a4fe41726ab2910153c5253df61e883771258db59440', 'repo_digests': ['vllm/vllm-openai@sha256:f21f5e1987142d4a7c77a4fe41726ab2910153c5253df61e883771258db59440']}`

## Results

| mode | concurrency | requests | output tok/s | req/s | p50 TTFT | p95 TTFT | p99 TTFT | p95 TPOT | p95 E2E | error | vs sequential | cost / 1M output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | 1 | 5 | 29.163 | 0.228 | 0.047 s | 0.047 s | 0.047 s | 0.034 s | 4.389 s | 0.00% | 1.00x | $9.58 |
| continuous | 8 | 40 | 209.360 | 1.636 | 0.102 s | 0.146 s | 0.147 s | 0.038 s | 4.896 s | 0.00% | 7.18x | $1.33 |
| continuous | 16 | 80 | 396.403 | 3.097 | 0.166 s | 0.201 s | 0.216 s | 0.039 s | 5.171 s | 0.00% | 13.59x | $0.70 |
| continuous | 32 | 160 | 739.868 | 5.780 | 0.222 s | 0.352 s | 0.365 s | 0.042 s | 5.537 s | 0.00% | 25.37x | $0.38 |

## Routing And Cache-Affinity Proxy

- Concurrency `1`: affinity-route rate `100.00%`, backend distribution `{'worker-0': 5}`, route reasons `{'prefix_affinity': 5}`.
- Concurrency `8`: affinity-route rate `100.00%`, backend distribution `{'worker-0': 40}`, route reasons `{'prefix_affinity': 40}`.
- Concurrency `16`: affinity-route rate `100.00%`, backend distribution `{'worker-0': 80}`, route reasons `{'prefix_affinity': 80}`.
- Concurrency `32`: affinity-route rate `100.00%`, backend distribution `{'worker-0': 160}`, route reasons `{'prefix_affinity': 160}`.

## GPU And vLLM Telemetry

- Concurrency `1` GPU: `{'0': {'sample_count': 41, 'mean_utilization_pct': 99.6829268292683, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 1984.0, 'prefix_cache_query_tokens': 2048.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.02650762094102055, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 1.0}`.
- Concurrency `8` GPU: `{'0': {'sample_count': 46, 'mean_utilization_pct': 99.6086956521739, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 15872.0, 'prefix_cache_query_tokens': 16384.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.12988734261100066, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 8.0}`.
- Concurrency `16` GPU: `{'0': {'sample_count': 49, 'mean_utilization_pct': 99.34693877551021, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 31744.0, 'prefix_cache_query_tokens': 32768.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.1776010603048377, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 16.0}`.
- Concurrency `32` GPU: `{'0': {'sample_count': 52, 'mean_utilization_pct': 98.98076923076923, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 63488.0, 'prefix_cache_query_tokens': 65536.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.27302849569251164, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 32.0}`.

## Method

- Concurrency 1 is the request-at-a-time sequential baseline; higher concurrency keeps the gateway continuously supplied.
- TTFT starts before the HTTP request and ends at the first non-empty generated token event.
- TPOT is `(E2E - TTFT) / (output_tokens - 1)` when at least two output tokens are observed.
- Output token counts use the streamed usage record when available and the configured tokenizer as a fallback.
- Cost per million output tokens uses measured wall time and the supplied instance hourly price; it excludes idle setup outside the measured windows.
- Prefix-affinity rate is a routing proxy, not a direct vLLM KV-cache hit rate. Direct cache usage is recorded separately when worker metrics expose it.
