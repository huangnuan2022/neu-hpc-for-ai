
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int seq_len;    // total sequence length
    int dim;        // head dimension
    int num_gpus;   // number of GPUs to use (1..8)
} FlashAttnConfig;

// Single GPU FlashAttention forward (reference / baseline).
// Q, K, V, and out are all device pointers on a single GPU.
// Shapes: [seq_len, dim]
void flash_attn_single_gpu_forward(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const FlashAttnConfig* cfg,
    cudaStream_t stream);

// Multi-GPU distributed FlashAttention forward using a ring-based
// sequence parallelism scheme.
// Each GPU owns seq_len / num_gpus query/key/value vectors.
// The d_q, d_k, d_v, d_out arrays are of length num_gpus;
// element i is a device pointer on GPU i with shape [local_seq_len, dim].
//
// For simplicity we assume seq_len is divisible by num_gpus and
// local_seq_len = seq_len / num_gpus.
void flash_attn_multi_gpu_forward(
    float** d_q,
    float** d_k,
    float** d_v,
    float** d_out,
    const FlashAttnConfig* cfg);

#ifdef __cplusplus
}
#endif
