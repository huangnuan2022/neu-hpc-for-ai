#include <math.h>
#include <stdio.h>

#include "flash_attn.h"
#include "utils.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define MAX_GPUS 8

__inline__ __device__ float warp_reduce_sum(float value)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__inline__ __device__ float block_reduce_sum(float value)
{
    static __shared__ float warp_sums[32];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    value = warp_reduce_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    const int warp_count = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
        value = warp_reduce_sum(value);
    }
    return value;
}

__global__ void init_flash_state_kernel(
    float* __restrict__ m,
    float* __restrict__ l,
    float* __restrict__ acc,
    int num_q,
    int dim)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_q) {
        m[index] = -INFINITY;
        l[index] = 0.0f;
    }
    const int total = num_q * dim;
    if (index < total) {
        acc[index] = 0.0f;
    }
}

__global__ void flash_attn_step_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_chunk,
    const float* __restrict__ v_chunk,
    float* __restrict__ acc,
    float* __restrict__ m,
    float* __restrict__ l,
    int num_q,
    int chunk_len,
    int dim,
    float scale)
{
    const int query = blockIdx.x;
    if (query >= num_q) {
        return;
    }

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* acc_shared = q_shared + dim;
    float* update = acc_shared + dim;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        q_shared[d] = q[(size_t)query * dim + d];
        acc_shared[d] = acc[(size_t)query * dim + d];
    }
    __syncthreads();

    float row_max = m[query];
    float row_sum = l[query];
    for (int key = 0; key < chunk_len; ++key) {
        float partial = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            partial += q_shared[d] * k_chunk[(size_t)key * dim + d];
        }

        float score = block_reduce_sum(partial);
        if (threadIdx.x == 0) {
            score *= scale;
            const float new_max = fmaxf(row_max, score);
            const float old_scale = expf(row_max - new_max);
            const float score_scale = expf(score - new_max);
            row_max = new_max;
            row_sum = row_sum * old_scale + score_scale;
            update[0] = old_scale;
            update[1] = score_scale;
        }
        __syncthreads();

        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            acc_shared[d] = acc_shared[d] * update[0]
                + v_chunk[(size_t)key * dim + d] * update[1];
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        acc[(size_t)query * dim + d] = acc_shared[d];
    }
    if (threadIdx.x == 0) {
        m[query] = row_max;
        l[query] = row_sum;
    }
}

__global__ void finalize_output_kernel(
    const float* __restrict__ acc,
    const float* __restrict__ l,
    float* __restrict__ out,
    int num_q,
    int dim)
{
    const int query = blockIdx.x;
    if (query >= num_q) {
        return;
    }
    const float denominator = l[query] > 0.0f ? l[query] : 1.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[(size_t)query * dim + d] = acc[(size_t)query * dim + d] / denominator;
    }
}

struct FlashAttnWorkspace {
    FlashAttnConfig cfg;
    int local_seq_len;
    size_t state_bytes;
    size_t chunk_bytes;
    size_t workspace_bytes_per_gpu;
    size_t measured_delta_bytes[MAX_GPUS];
    ncclComm_t comms[MAX_GPUS];
    cudaStream_t compute_streams[MAX_GPUS];
    cudaStream_t communication_streams[MAX_GPUS];
    cudaEvent_t start_events[MAX_GPUS];
    cudaEvent_t stop_events[MAX_GPUS];
    cudaEvent_t compute_done[MAX_GPUS][2];
    cudaEvent_t receive_ready[MAX_GPUS][2];
    float* d_m[MAX_GPUS];
    float* d_l[MAX_GPUS];
    float* d_acc[MAX_GPUS];
    float* d_k_ring[MAX_GPUS][2];
    float* d_v_ring[MAX_GPUS][2];
};

static void validate_config(const FlashAttnConfig* cfg)
{
    if (cfg == NULL || cfg->seq_len <= 0 || cfg->dim <= 0
        || cfg->num_gpus <= 0 || cfg->num_gpus > MAX_GPUS) {
        fprintf(stderr, "Invalid FlashAttnConfig\n");
        exit(EXIT_FAILURE);
    }
    if (cfg->seq_len % cfg->num_gpus != 0) {
        fprintf(stderr, "Sequence length %d must be divisible by %d GPUs\n",
                cfg->seq_len, cfg->num_gpus);
        exit(EXIT_FAILURE);
    }
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count < cfg->num_gpus) {
        fprintf(stderr, "Requested %d GPUs but only %d are available\n",
                cfg->num_gpus, device_count);
        exit(EXIT_FAILURE);
    }
}

FlashAttnWorkspace* flash_attn_workspace_create(const FlashAttnConfig* cfg)
{
    validate_config(cfg);
    FlashAttnWorkspace* workspace = new FlashAttnWorkspace();
    workspace->cfg = *cfg;
    workspace->local_seq_len = cfg->seq_len / cfg->num_gpus;
    workspace->state_bytes = (size_t)workspace->local_seq_len * sizeof(float);
    workspace->chunk_bytes = (size_t)workspace->local_seq_len * cfg->dim * sizeof(float);
    workspace->workspace_bytes_per_gpu = 2 * workspace->state_bytes
        + workspace->chunk_bytes
        + 4 * workspace->chunk_bytes;

    int devices[MAX_GPUS];
    for (int gpu = 0; gpu < cfg->num_gpus; ++gpu) {
        devices[gpu] = gpu;
    }
    if (cfg->num_gpus > 1) {
        CHECK_NCCL(ncclCommInitAll(workspace->comms, cfg->num_gpus, devices));
    }

    for (int gpu = 0; gpu < cfg->num_gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamCreateWithFlags(&workspace->compute_streams[gpu], cudaStreamNonBlocking));
        CHECK_CUDA(cudaStreamCreateWithFlags(
            &workspace->communication_streams[gpu], cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&workspace->start_events[gpu]));
        CHECK_CUDA(cudaEventCreate(&workspace->stop_events[gpu]));
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_CUDA(cudaEventCreateWithFlags(
                &workspace->compute_done[gpu][slot], cudaEventDisableTiming));
            CHECK_CUDA(cudaEventCreateWithFlags(
                &workspace->receive_ready[gpu][slot], cudaEventDisableTiming));
        }
        size_t free_before = 0;
        size_t total_before = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_before, &total_before));
        CHECK_CUDA(cudaMalloc(&workspace->d_m[gpu], workspace->state_bytes));
        CHECK_CUDA(cudaMalloc(&workspace->d_l[gpu], workspace->state_bytes));
        CHECK_CUDA(cudaMalloc(&workspace->d_acc[gpu], workspace->chunk_bytes));
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_CUDA(cudaMalloc(&workspace->d_k_ring[gpu][slot], workspace->chunk_bytes));
            CHECK_CUDA(cudaMalloc(&workspace->d_v_ring[gpu][slot], workspace->chunk_bytes));
        }
        size_t free_after = 0;
        size_t total_after = 0;
        CHECK_CUDA(cudaMemGetInfo(&free_after, &total_after));
        workspace->measured_delta_bytes[gpu] = free_before >= free_after
            ? free_before - free_after : 0;
    }
    return workspace;
}

void flash_attn_workspace_prepare_kv(
    FlashAttnWorkspace* workspace,
    float* const* d_k,
    float* const* d_v)
{
    for (int gpu = 0; gpu < workspace->cfg.num_gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaMemcpyAsync(
            workspace->d_k_ring[gpu][0], d_k[gpu], workspace->chunk_bytes,
            cudaMemcpyDeviceToDevice, workspace->communication_streams[gpu]));
        CHECK_CUDA(cudaMemcpyAsync(
            workspace->d_v_ring[gpu][0], d_v[gpu], workspace->chunk_bytes,
            cudaMemcpyDeviceToDevice, workspace->communication_streams[gpu]));
    }
    for (int gpu = 0; gpu < workspace->cfg.num_gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaStreamSynchronize(workspace->communication_streams[gpu]));
    }
}

double flash_attn_workspace_forward(
    FlashAttnWorkspace* workspace,
    float* const* d_q,
    float* const* d_out)
{
    const int gpu_count = workspace->cfg.num_gpus;
    const int local_seq = workspace->local_seq_len;
    const int dim = workspace->cfg.dim;
    const size_t chunk_elements = (size_t)local_seq * dim;
    const int state_threads = 256;
    const int state_blocks = (local_seq * dim + state_threads - 1) / state_threads;
    const int attention_threads = 128;
    const size_t shared_bytes = (size_t)(2 * dim + 2) * sizeof(float);
    const float scale = 1.0f / sqrtf((float)dim);

    for (int gpu = 0; gpu < gpu_count; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaEventRecord(
            workspace->start_events[gpu], workspace->compute_streams[gpu]));
        CHECK_CUDA(cudaStreamWaitEvent(
            workspace->communication_streams[gpu], workspace->start_events[gpu], 0));
        init_flash_state_kernel<<<
            state_blocks, state_threads, 0, workspace->compute_streams[gpu]>>>(
                workspace->d_m[gpu], workspace->d_l[gpu], workspace->d_acc[gpu],
                local_seq, dim);
        CHECK_CUDA(cudaGetLastError());
    }

    for (int step = 0; step < gpu_count; ++step) {
        const int current_slot = step % 2;
        const int next_slot = 1 - current_slot;

        for (int gpu = 0; gpu < gpu_count; ++gpu) {
            CHECK_CUDA(cudaSetDevice(gpu));
            if (step > 0) {
                CHECK_CUDA(cudaStreamWaitEvent(
                    workspace->compute_streams[gpu],
                    workspace->receive_ready[gpu][current_slot], 0));
            }
            flash_attn_step_kernel<<<
                local_seq, attention_threads, shared_bytes,
                workspace->compute_streams[gpu]>>>(
                    d_q[gpu], workspace->d_k_ring[gpu][current_slot],
                    workspace->d_v_ring[gpu][current_slot], workspace->d_acc[gpu],
                    workspace->d_m[gpu], workspace->d_l[gpu], local_seq, local_seq,
                    dim, scale);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaEventRecord(
                workspace->compute_done[gpu][current_slot],
                workspace->compute_streams[gpu]));
        }

        if (step >= gpu_count - 1) {
            continue;
        }

        if (!workspace->cfg.overlap_kv_rotation) {
            for (int gpu = 0; gpu < gpu_count; ++gpu) {
                CHECK_CUDA(cudaSetDevice(gpu));
                CHECK_CUDA(cudaStreamWaitEvent(
                    workspace->communication_streams[gpu],
                    workspace->compute_done[gpu][current_slot], 0));
            }
        }
        if (step >= 1) {
            for (int gpu = 0; gpu < gpu_count; ++gpu) {
                CHECK_CUDA(cudaSetDevice(gpu));
                CHECK_CUDA(cudaStreamWaitEvent(
                    workspace->communication_streams[gpu],
                    workspace->compute_done[gpu][next_slot], 0));
            }
        }

        CHECK_NCCL(ncclGroupStart());
        for (int gpu = 0; gpu < gpu_count; ++gpu) {
            const int next_gpu = (gpu + 1) % gpu_count;
            const int previous_gpu = (gpu - 1 + gpu_count) % gpu_count;
            CHECK_CUDA(cudaSetDevice(gpu));
            CHECK_NCCL(ncclSend(
                workspace->d_k_ring[gpu][current_slot], chunk_elements, ncclFloat,
                next_gpu, workspace->comms[gpu], workspace->communication_streams[gpu]));
            CHECK_NCCL(ncclRecv(
                workspace->d_k_ring[gpu][next_slot], chunk_elements, ncclFloat,
                previous_gpu, workspace->comms[gpu], workspace->communication_streams[gpu]));
            CHECK_NCCL(ncclSend(
                workspace->d_v_ring[gpu][current_slot], chunk_elements, ncclFloat,
                next_gpu, workspace->comms[gpu], workspace->communication_streams[gpu]));
            CHECK_NCCL(ncclRecv(
                workspace->d_v_ring[gpu][next_slot], chunk_elements, ncclFloat,
                previous_gpu, workspace->comms[gpu], workspace->communication_streams[gpu]));
        }
        CHECK_NCCL(ncclGroupEnd());
        for (int gpu = 0; gpu < gpu_count; ++gpu) {
            CHECK_CUDA(cudaSetDevice(gpu));
            CHECK_CUDA(cudaEventRecord(
                workspace->receive_ready[gpu][next_slot],
                workspace->communication_streams[gpu]));
        }
    }

    for (int gpu = 0; gpu < gpu_count; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        finalize_output_kernel<<<
            local_seq, attention_threads, 0, workspace->compute_streams[gpu]>>>(
                workspace->d_acc[gpu], workspace->d_l[gpu], d_out[gpu],
                local_seq, dim);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(
            workspace->stop_events[gpu], workspace->compute_streams[gpu]));
    }

    double max_elapsed_ms = 0.0;
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaEventSynchronize(workspace->stop_events[gpu]));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(
            &elapsed_ms, workspace->start_events[gpu], workspace->stop_events[gpu]));
        if (elapsed_ms > max_elapsed_ms) {
            max_elapsed_ms = elapsed_ms;
        }
    }
    return max_elapsed_ms;
}

void flash_attn_workspace_destroy(FlashAttnWorkspace* workspace)
{
    if (workspace == NULL) {
        return;
    }
    for (int gpu = 0; gpu < workspace->cfg.num_gpus; ++gpu) {
        CHECK_CUDA(cudaSetDevice(gpu));
        CHECK_CUDA(cudaFree(workspace->d_m[gpu]));
        CHECK_CUDA(cudaFree(workspace->d_l[gpu]));
        CHECK_CUDA(cudaFree(workspace->d_acc[gpu]));
        for (int slot = 0; slot < 2; ++slot) {
            CHECK_CUDA(cudaFree(workspace->d_k_ring[gpu][slot]));
            CHECK_CUDA(cudaFree(workspace->d_v_ring[gpu][slot]));
            CHECK_CUDA(cudaEventDestroy(workspace->compute_done[gpu][slot]));
            CHECK_CUDA(cudaEventDestroy(workspace->receive_ready[gpu][slot]));
        }
        CHECK_CUDA(cudaEventDestroy(workspace->start_events[gpu]));
        CHECK_CUDA(cudaEventDestroy(workspace->stop_events[gpu]));
        CHECK_CUDA(cudaStreamDestroy(workspace->compute_streams[gpu]));
        CHECK_CUDA(cudaStreamDestroy(workspace->communication_streams[gpu]));
        if (workspace->cfg.num_gpus > 1) {
            CHECK_NCCL(ncclCommDestroy(workspace->comms[gpu]));
        }
    }
    delete workspace;
}

size_t flash_attn_workspace_bytes_per_gpu(const FlashAttnWorkspace* workspace)
{
    return workspace->workspace_bytes_per_gpu;
}

size_t flash_attn_workspace_measured_delta_bytes(
    const FlashAttnWorkspace* workspace,
    int gpu_index)
{
    if (gpu_index < 0 || gpu_index >= workspace->cfg.num_gpus) {
        return 0;
    }
    return workspace->measured_delta_bytes[gpu_index];
}

size_t flash_attn_minimal_state_bytes_per_gpu(const FlashAttnConfig* cfg)
{
    const size_t local_seq = (size_t)cfg->seq_len / cfg->num_gpus;
    const size_t state = 2 * local_seq * sizeof(float);
    const size_t accumulator = local_seq * cfg->dim * sizeof(float);
    const size_t current_kv_shard = 2 * local_seq * cfg->dim * sizeof(float);
    return state + accumulator + current_kv_shard;
}

size_t flash_attn_full_score_matrix_bytes(const FlashAttnConfig* cfg)
{
    return (size_t)cfg->seq_len * cfg->seq_len * sizeof(float);
}
