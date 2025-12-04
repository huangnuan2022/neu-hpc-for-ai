
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "flash_attn.h"
#include "utils.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Simple block-wide reduction for float sums.
__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val)
{
    static __shared__ float shared[32]; // max 32 warps per block
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val); // each warp reduces to lane 0

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // only first warp loads the warp results
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

// Initialize streaming softmax state for all local queries.
__global__ void init_flash_state_kernel(
    float* __restrict__ m,
    float* __restrict__ l,
    float* __restrict__ acc,
    int num_q,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // m and l: one per query
    if (idx < num_q) {
        m[idx] = -1e30f; // "minus infinity" sentinel
        l[idx] = 0.0f;
    }

    // acc: num_q * dim elements
    int total = num_q * dim;
    if (idx < total) {
        acc[idx] = 0.0f;
    }
}

// One streaming FlashAttention step over a chunk of keys/values.
//
// Each block processes a single query vector, computing its interaction
// with chunk_len keys. The kernel maintains a numerically stable running
// softmax using the usual (m, l) trick:
//
//   m_new = max(m, s)
//   l_new = l * exp(m - m_new) + exp(s - m_new)
//   acc   = acc * exp(m - m_new) + V * exp(s - m_new)
//
// After all chunks have been processed, acc / l is the final output.
__global__ void flash_attn_step_kernel(
    const float* __restrict__ q,         // [num_q, dim]
    const float* __restrict__ k_chunk,   // [chunk_len, dim]
    const float* __restrict__ v_chunk,   // [chunk_len, dim]
    float* __restrict__ acc,             // [num_q, dim]
    float* __restrict__ m,               // [num_q]
    float* __restrict__ l,               // [num_q]
    int num_q,
    int chunk_len,
    int dim,
    float scale)
{
    int q_idx = blockIdx.x;
    if (q_idx >= num_q) return;

    extern __shared__ float shared[];
    float* q_sh   = shared;          // dim floats
    float* acc_sh = q_sh + dim;      // dim floats
    float* tmp    = acc_sh + dim;    // at least 2 floats

    // Load query into shared memory
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        q_sh[d] = q[q_idx * dim + d];
    }

    // Load current acc for this query into shared
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc_sh[d] = acc[q_idx * dim + d];
    }

    __syncthreads();

    float m_i = m[q_idx];
    float l_i = l[q_idx];

    for (int key_idx = 0; key_idx < chunk_len; ++key_idx) {
        // Compute dot(q_i, k_j)
        float thread_sum = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float qv = q_sh[d];
            float kv = k_chunk[key_idx * dim + d];
            thread_sum += qv * kv;
        }

        float score = block_reduce_sum(thread_sum);
        if (threadIdx.x == 0) {
            score *= scale;

            float m_new = fmaxf(m_i, score);
            float exp_m = __expf(m_i - m_new);
            float exp_s = __expf(score - m_new);

            m_i = m_new;
            l_i = l_i * exp_m + exp_s;

            tmp[0] = exp_m;
            tmp[1] = exp_s;
        }

        __syncthreads();

        float exp_m = tmp[0];
        float exp_s = tmp[1];

        // Update accumulator vector
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float vval = v_chunk[key_idx * dim + d];
            acc_sh[d] = acc_sh[d] * exp_m + vval * exp_s;
        }

        __syncthreads();
    }

    // Write back updated accumulator and state
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc[q_idx * dim + d] = acc_sh[d];
    }

    if (threadIdx.x == 0) {
        m[q_idx] = m_i;
        l[q_idx] = l_i;
    }
}

// Finalize outputs: out = acc / l
__global__ void finalize_output_kernel(
    const float* __restrict__ acc,
    const float* __restrict__ l,
    float* __restrict__ out,
    int num_q,
    int dim)
{
    int q_idx = blockIdx.x;
    if (q_idx >= num_q) return;

    float norm = l[q_idx];
    if (norm <= 0.0f) norm = 1.0f;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = acc[q_idx * dim + d];
        out[q_idx * dim + d] = v / norm;
    }
}

/******************************
 * Single GPU implementation  *
 ******************************/

void flash_attn_single_gpu_forward(
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_out,
    const FlashAttnConfig* cfg,
    cudaStream_t stream)
{
    int seq_len = cfg->seq_len;
    int dim     = cfg->dim;

    int num_q     = seq_len;
    int chunk_len = seq_len;

    float* d_m   = NULL;
    float* d_l   = NULL;
    float* d_acc = NULL;

    size_t state_bytes = num_q * sizeof(float);
    size_t acc_bytes   = (size_t)num_q * dim * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_m,   state_bytes));
    CHECK_CUDA(cudaMalloc(&d_l,   state_bytes));
    CHECK_CUDA(cudaMalloc(&d_acc, acc_bytes));

    // Initialize state
    int block = 256;
    int grid_state = (num_q * dim + block - 1) / block;
    init_flash_state_kernel<<<grid_state, block, 0, stream>>>(
        d_m, d_l, d_acc, num_q, dim);
    CHECK_CUDA(cudaGetLastError());

    float scale = 1.0f / sqrtf((float)dim);

    dim3 grid_q(num_q);
    int  block_q = 128;
    size_t shared_bytes = (2 * dim + 2) * sizeof(float);

    flash_attn_step_kernel<<<grid_q, block_q, shared_bytes, stream>>>(
        d_q, d_k, d_v, d_acc, d_m, d_l,
        num_q, chunk_len, dim, scale);
    CHECK_CUDA(cudaGetLastError());

    finalize_output_kernel<<<grid_q, block_q, 0, stream>>>(
        d_acc, d_l, d_out, num_q, dim);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFree(d_m));
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_acc));
}

/******************************
 * Multi-GPU implementation   *
 ******************************/

void flash_attn_multi_gpu_forward(
    float** d_q,
    float** d_k,
    float** d_v,
    float** d_out,
    const FlashAttnConfig* cfg)
{
    int num_gpus = cfg->num_gpus;
    int seq_len  = cfg->seq_len;
    int dim      = cfg->dim;

    if (num_gpus <= 0) {
        fprintf(stderr, "flash_attn_multi_gpu_forward: num_gpus must be > 0\n");
        exit(EXIT_FAILURE);
    }

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < num_gpus) {
        fprintf(stderr, "Requested %d GPUs but only %d available\n",
                num_gpus, device_count);
        exit(EXIT_FAILURE);
    }

    if (seq_len % num_gpus != 0) {
        fprintf(stderr, "Sequence length %d must be divisible by num_gpus %d\n",
                seq_len, num_gpus);
        exit(EXIT_FAILURE);
    }

    int local_seq_len = seq_len / num_gpus;
    int devs[8];
    for (int i = 0; i < num_gpus; ++i) {
        devs[i] = i;
    }

    ncclComm_t comms[8];
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, devs));

    cudaStream_t streams[8];
    float* d_m[8]      = {0};
    float* d_l[8]      = {0};
    float* d_acc[8]    = {0};
    float* d_k_buf[8]  = {0};
    float* d_v_buf[8]  = {0};
    float* cur_k[8]    = {0};
    float* cur_v[8]    = {0};

    size_t state_bytes = local_seq_len * sizeof(float);
    size_t chunk_elems = (size_t)local_seq_len * dim;
    size_t chunk_bytes = chunk_elems * sizeof(float);

    // Per-GPU initialization
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(devs[i]));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));

        CHECK_CUDA(cudaMalloc(&d_m[i], state_bytes));
        CHECK_CUDA(cudaMalloc(&d_l[i], state_bytes));
        CHECK_CUDA(cudaMalloc(&d_acc[i], chunk_bytes));
        CHECK_CUDA(cudaMalloc(&d_k_buf[i], chunk_bytes));
        CHECK_CUDA(cudaMalloc(&d_v_buf[i], chunk_bytes));

        int block = 256;
        int grid_state = (local_seq_len * dim + block - 1) / block;
        init_flash_state_kernel<<<grid_state, block, 0, streams[i]>>>(
            d_m[i], d_l[i], d_acc[i], local_seq_len, dim);
        CHECK_CUDA(cudaGetLastError());

        cur_k[i] = d_k[i];
        cur_v[i] = d_v[i];
    }

    float scale = 1.0f / sqrtf((float)dim);
    dim3 grid_q(local_seq_len);
    int  block_q = 128;
    size_t shared_bytes = (2 * dim + 2) * sizeof(float);

    // Ring over key/value shards
    for (int step = 0; step < num_gpus; ++step) {
        // Local compute
        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA(cudaSetDevice(devs[i]));
            flash_attn_step_kernel<<<grid_q, block_q, shared_bytes, streams[i]>>>(
                d_q[i], cur_k[i], cur_v[i],
                d_acc[i], d_m[i], d_l[i],
                local_seq_len, local_seq_len, dim, scale);
            CHECK_CUDA(cudaGetLastError());
        }

        if (step < num_gpus - 1) {
            // Rotate K/V in a ring: i -> (i+1) mod N
            CHECK_NCCL(ncclGroupStart());
            for (int i = 0; i < num_gpus; ++i) {
                int next = (i + 1) % num_gpus;
                int prev = (i - 1 + num_gpus) % num_gpus;

                CHECK_CUDA(cudaSetDevice(devs[i]));
                CHECK_NCCL(ncclSend(cur_k[i], chunk_elems, ncclFloat,
                                    next, comms[i], streams[i]));
                CHECK_NCCL(ncclRecv(d_k_buf[i], chunk_elems, ncclFloat,
                                    prev, comms[i], streams[i]));
                CHECK_NCCL(ncclSend(cur_v[i], chunk_elems, ncclFloat,
                                    next, comms[i], streams[i]));
                CHECK_NCCL(ncclRecv(d_v_buf[i], chunk_elems, ncclFloat,
                                    prev, comms[i], streams[i]));
            }
            CHECK_NCCL(ncclGroupEnd());

            // Swap buffers for next step
            for (int i = 0; i < num_gpus; ++i) {
                float* tmp_k = cur_k[i];
                float* tmp_v = cur_v[i];
                cur_k[i] = d_k_buf[i];
                cur_v[i] = d_v_buf[i];
                d_k_buf[i] = tmp_k;
                d_v_buf[i] = tmp_v;
            }
        }
    }

    // Finalize outputs
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(devs[i]));
        finalize_output_kernel<<<grid_q, block_q, 0, streams[i]>>>(
            d_acc[i], d_l[i], d_out[i],
            local_seq_len, dim);
        CHECK_CUDA(cudaGetLastError());
    }

    // Sync & cleanup
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(devs[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA(cudaFree(d_m[i]));
        CHECK_CUDA(cudaFree(d_l[i]));
        CHECK_CUDA(cudaFree(d_acc[i]));
        CHECK_CUDA(cudaFree(d_k_buf[i]));
        CHECK_CUDA(cudaFree(d_v_buf[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_NCCL(ncclCommDestroy(comms[i]));
    }
}
