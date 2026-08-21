# Serving Benchmark Report

- Evidence status: **Measured vLLM performance evidence.**
- Generated at: `2026-08-21T09:36:30.036754+00:00`
- Deployment: `aws-a10g-4gpu`
- Model: `Qwen/Qwen3-8B`
- Backend: `vllm` with `4` worker(s)
- Workload: `512` input tokens, `128` requested output tokens
- Concurrency: `[1, 8, 16, 32]`
- Runs: `1` warm-up + `5` measured per concurrency
- Git SHA: `e056e29e326868f7a88405f6ae13b3a02f2d3bf7` (dirty: `False`)
- AWS: `{'instance_id': 'i-0108995b4a93240a1', 'instance_type': 'g5.12xlarge', 'ami_id': 'ami-0fe19eeadb42a3627', 'region': 'us-east-1'}`
- GPUs: `[{'index': 0, 'name': 'NVIDIA A10G', 'driver_version': '595.91.07', 'memory_total_mib': 23028.0}, {'index': 1, 'name': 'NVIDIA A10G', 'driver_version': '595.91.07', 'memory_total_mib': 23028.0}, {'index': 2, 'name': 'NVIDIA A10G', 'driver_version': '595.91.07', 'memory_total_mib': 23028.0}, {'index': 3, 'name': 'NVIDIA A10G', 'driver_version': '595.91.07', 'memory_total_mib': 23028.0}]`
- CUDA toolkit: `13.3`
- CUDA driver API: `13.2`
- vLLM image: `{'requested': 'vllm/vllm-openai:v0.26.0-cu129-ubuntu2404', 'image_id': 'sha256:f21f5e1987142d4a7c77a4fe41726ab2910153c5253df61e883771258db59440', 'repo_digests': ['vllm/vllm-openai@sha256:f21f5e1987142d4a7c77a4fe41726ab2910153c5253df61e883771258db59440']}`

## Results

| mode | concurrency | requests | output tok/s | req/s | p50 TTFT | p95 TTFT | p99 TTFT | p95 TPOT | p95 E2E | error | vs sequential | cost / 1M output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | 1 | 5 | 29.167 | 0.228 | 0.046 s | 0.049 s | 0.050 s | 0.034 s | 4.389 s | 0.00% | 1.00x | $54.02 |
| continuous | 8 | 40 | 216.926 | 1.695 | 0.087 s | 0.123 s | 0.146 s | 0.036 s | 4.722 s | 0.00% | 7.44x | $7.26 |
| continuous | 16 | 80 | 424.329 | 3.315 | 0.108 s | 0.146 s | 0.148 s | 0.037 s | 4.824 s | 0.00% | 14.55x | $3.71 |
| continuous | 32 | 160 | 786.408 | 6.144 | 0.175 s | 0.243 s | 0.252 s | 0.039 s | 5.195 s | 0.00% | 26.96x | $2.00 |

## Routing And Cache-Affinity Proxy

- Concurrency `1`: affinity-route rate `100.00%`, backend distribution `{'worker-2': 5}`, route reasons `{'prefix_affinity': 5}`.
- Concurrency `8`: affinity-route rate `87.50%`, backend distribution `{'worker-0': 15, 'worker-1': 10, 'worker-2': 10, 'worker-3': 5}`, route reasons `{'affinity_overridden_by_load': 5, 'prefix_affinity': 35}`.
- Concurrency `16`: affinity-route rate `81.25%`, backend distribution `{'worker-0': 25, 'worker-1': 20, 'worker-2': 20, 'worker-3': 15}`, route reasons `{'affinity_overridden_by_load': 15, 'prefix_affinity': 65}`.
- Concurrency `32`: affinity-route rate `75.00%`, backend distribution `{'worker-0': 45, 'worker-1': 45, 'worker-2': 35, 'worker-3': 35}`, route reasons `{'affinity_overridden_by_load': 40, 'prefix_affinity': 120}`.

## GPU And vLLM Telemetry

- Concurrency `1` GPU: `{'0': {'sample_count': 38, 'mean_utilization_pct': 0.0, 'max_utilization_pct': 0.0, 'max_memory_used_mib': 20797.0}, '1': {'sample_count': 38, 'mean_utilization_pct': 0.0, 'max_utilization_pct': 0.0, 'max_memory_used_mib': 20797.0}, '2': {'sample_count': 38, 'mean_utilization_pct': 99.78947368421052, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '3': {'sample_count': 38, 'mean_utilization_pct': 0.0, 'max_utilization_pct': 0.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 1984.0, 'prefix_cache_query_tokens': 2048.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.02650762094102055, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 1.0}`.
- Concurrency `8` GPU: `{'0': {'sample_count': 40, 'mean_utilization_pct': 100.0, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '1': {'sample_count': 40, 'mean_utilization_pct': 100.0, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '2': {'sample_count': 40, 'mean_utilization_pct': 100.0, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '3': {'sample_count': 40, 'mean_utilization_pct': 90.35, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 15872.0, 'prefix_cache_query_tokens': 16384.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.058979456593770685, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 8.0}`.
- Concurrency `16` GPU: `{'0': {'sample_count': 41, 'mean_utilization_pct': 98.6829268292683, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '1': {'sample_count': 41, 'mean_utilization_pct': 98.17073170731707, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '2': {'sample_count': 41, 'mean_utilization_pct': 98.09756097560975, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '3': {'sample_count': 41, 'mean_utilization_pct': 96.39024390243902, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 31744.0, 'prefix_cache_query_tokens': 32768.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.07090788601722997, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 16.0}`.
- Concurrency `32` GPU: `{'0': {'sample_count': 44, 'mean_utilization_pct': 98.9090909090909, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '1': {'sample_count': 44, 'mean_utilization_pct': 98.70454545454545, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '2': {'sample_count': 44, 'mean_utilization_pct': 92.68181818181819, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}, '3': {'sample_count': 44, 'mean_utilization_pct': 93.22727272727273, 'max_utilization_pct': 100.0, 'max_memory_used_mib': 20797.0}}`; vLLM: `{'prefix_cache_hit_tokens': 63488.0, 'prefix_cache_query_tokens': 65536.0, 'prefix_cache_hit_rate': 0.96875, 'max_kv_cache_usage': 0.1033797216699801, 'max_requests_waiting_sum': 0.0, 'max_requests_running_sum': 32.0}`.

## Method

- Concurrency 1 is the request-at-a-time sequential baseline; higher concurrency keeps the gateway continuously supplied.
- TTFT starts before the HTTP request and ends at the first non-empty generated token event.
- TPOT is `(E2E - TTFT) / (output_tokens - 1)` when at least two output tokens are observed.
- Output token counts use the streamed usage record when available and the configured tokenizer as a fallback.
- Cost per million output tokens uses measured wall time and the supplied instance hourly price; it excludes idle setup outside the measured windows.
- Prefix-affinity rate is a routing proxy, not a direct vLLM KV-cache hit rate. Direct cache usage is recorded separately when worker metrics expose it.
