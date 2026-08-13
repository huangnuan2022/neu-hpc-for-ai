# Benchmark Summary

- Generated at: `2026-08-13T05:50:25.181608+00:00`
- Host: `macOS-14.7.6-arm64-arm-64bit-Mach-O`
- Compiler binary: `build/inference_harness`

## Key Results

- Verified streaming and ring-sequence-parallel attention against a materialized softmax reference across `4` shape sweeps.
- Verified routed expert-sharded MoE execution against a dense top-1 reference across `3` shape sweeps.
- Maximum attention correctness drift: `0.000e+00`.
- Maximum MoE correctness drift: `0.000e+00`.
- Largest attention sweep: `seq=1024`, `dim=64`, `shards=4`.
- Estimated peak per-shard attention working-memory reduction versus materializing the score matrix: `95.26%`.

## Attention Sweeps

| seq | dim | shards | naive | streaming | ring sim | max ring diff | memory reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 2 | 0.616 ms | 0.325 ms | 0.329 ms | 0.000e+00 | 61.72% |
| 256 | 64 | 4 | 5.733 ms | 2.847 ms | 2.867 ms | 0.000e+00 | 81.05% |
| 512 | 64 | 4 | 25.450 ms | 11.457 ms | 11.634 ms | 0.000e+00 | 90.53% |
| 1024 | 64 | 4 | 125.168 ms | 46.039 ms | 46.461 ms | 0.000e+00 | 95.26% |

## MoE Routing Sweeps

| tokens | dim | hidden | experts | shards | dense | routed sim | max diff | route imbalance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 64 | 8 | 4 | 0.340 ms | 0.356 ms | 0.000e+00 | 2.05x |
| 256 | 32 | 64 | 8 | 4 | 0.653 ms | 0.652 ms | 0.000e+00 | 1.42x |
| 512 | 32 | 64 | 8 | 4 | 1.364 ms | 1.297 ms | 0.000e+00 | 1.40x |

## Resume-Safe Claims

- Correctness-gated `7` benchmark sweeps for attention and MoE inference primitives.
- Attention ring simulator preserved output parity within `0.000e+00` max absolute error.
- Routed MoE simulator preserved output parity within `0.000e+00` max absolute error on the tested shapes.
- Largest measured attention shape avoided materializing a `1024 x 1024` score matrix in the streaming/ring path.
- Largest MoE sweep routed `512` tokens across `8` experts and `4` logical shards.
