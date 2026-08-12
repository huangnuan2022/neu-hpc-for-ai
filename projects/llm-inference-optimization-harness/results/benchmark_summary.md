# Benchmark Summary

- Generated at: `2026-08-12T01:00:08.023723+00:00`
- Host: `macOS-14.7.6-arm64-arm-64bit`
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
| 128 | 32 | 2 | 0.584 ms | 0.316 ms | 0.313 ms | 0.000e+00 | 61.72% |
| 256 | 64 | 4 | 5.603 ms | 2.717 ms | 2.779 ms | 0.000e+00 | 81.05% |
| 512 | 64 | 4 | 24.832 ms | 11.128 ms | 11.092 ms | 0.000e+00 | 90.53% |
| 1024 | 64 | 4 | 103.853 ms | 43.399 ms | 44.067 ms | 0.000e+00 | 95.26% |

## MoE Routing Sweeps

| tokens | dim | hidden | experts | shards | dense | routed sim | max diff | route imbalance |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 64 | 8 | 4 | 0.321 ms | 0.330 ms | 0.000e+00 | 2.05x |
| 256 | 32 | 64 | 8 | 4 | 0.662 ms | 0.659 ms | 0.000e+00 | 1.42x |
| 512 | 32 | 64 | 8 | 4 | 1.299 ms | 1.308 ms | 0.000e+00 | 1.40x |

## Resume-Safe Claims

- Correctness-gated `7` benchmark sweeps for attention and MoE inference primitives.
- Attention ring simulator preserved output parity within `0.000e+00` max absolute error.
- Routed MoE simulator preserved output parity within `0.000e+00` max absolute error on the tested shapes.
- Largest measured attention shape avoided materializing a `1024 x 1024` score matrix in the streaming/ring path.
- Largest MoE sweep routed `512` tokens across `8` experts and `4` logical shards.
