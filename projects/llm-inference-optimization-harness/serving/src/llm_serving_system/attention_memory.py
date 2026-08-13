from __future__ import annotations


FP32_BYTES = 4


def attention_memory(seq: int, dim: int, gpus: int) -> dict[str, float | int]:
    if seq <= 0 or dim <= 0 or gpus <= 0 or seq % gpus:
        raise ValueError("positive seq and dim are required, and gpus must divide seq")
    local_seq = seq // gpus
    full_score = seq * seq * FP32_BYTES
    state = 2 * local_seq * FP32_BYTES
    accumulator = local_seq * dim * FP32_BYTES
    current_kv = 2 * local_seq * dim * FP32_BYTES
    second_kv_buffer = current_kv
    minimal = state + accumulator + current_kv
    explicit_double_buffer = minimal + second_kv_buffer
    return {
        "full_score_matrix_bytes": full_score,
        "minimal_ring_state_bytes_per_gpu": minimal,
        "explicit_double_buffer_workspace_bytes_per_gpu": explicit_double_buffer,
        "minimal_state_reduction_pct": 100.0 * (1.0 - minimal / full_score),
        "explicit_workspace_reduction_pct": 100.0 * (1.0 - explicit_double_buffer / full_score),
    }
