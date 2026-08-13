#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int seq_len;
    int dim;
    int num_gpus;
    int overlap_kv_rotation;
} FlashAttnConfig;

typedef struct FlashAttnWorkspace FlashAttnWorkspace;

FlashAttnWorkspace* flash_attn_workspace_create(const FlashAttnConfig* cfg);

void flash_attn_workspace_destroy(FlashAttnWorkspace* workspace);

// Stages the local K/V shards into reusable internal ring buffers. This is
// intentionally outside the steady-state timed forward pass.
void flash_attn_workspace_prepare_kv(
    FlashAttnWorkspace* workspace,
    float* const* d_k,
    float* const* d_v);

// Executes state initialization, streaming attention, NCCL K/V rotation, and
// output finalization. Returns the maximum per-device CUDA-event duration.
double flash_attn_workspace_forward(
    FlashAttnWorkspace* workspace,
    float* const* d_q,
    float* const* d_out);

size_t flash_attn_workspace_bytes_per_gpu(const FlashAttnWorkspace* workspace);

size_t flash_attn_workspace_measured_delta_bytes(
    const FlashAttnWorkspace* workspace,
    int gpu_index);

size_t flash_attn_minimal_state_bytes_per_gpu(const FlashAttnConfig* cfg);

size_t flash_attn_full_score_matrix_bytes(const FlashAttnConfig* cfg);

#ifdef __cplusplus
}
#endif
