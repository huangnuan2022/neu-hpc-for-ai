
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "flash_attn.h"
#include "utils.h"

static void parse_args(int argc, char** argv,
                       int* seq_len, int* dim, int* num_gpus)
{
    *seq_len = 1024;
    *dim     = 64;
    *num_gpus = 2;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--seq") && i + 1 < argc) {
            *seq_len = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--dim") && i + 1 < argc) {
            *dim = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gpus") && i + 1 < argc) {
            *num_gpus = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            printf("Usage: %s [--seq L] [--dim D] [--gpus N]\\n", argv[0]);
            exit(0);
        }
    }
}

int main(int argc, char** argv)
{
    int seq_len, dim, num_gpus;
    parse_args(argc, argv, &seq_len, &dim, &num_gpus);

    printf("Config: seq_len=%d dim=%d num_gpus=%d\\n",
           seq_len, dim, num_gpus);

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        fprintf(stderr, "No CUDA devices found.\\n");
        return EXIT_FAILURE;
    }

    if (num_gpus > device_count) {
        fprintf(stderr, "Requested %d GPUs but only %d available, clamping.\\n",
                num_gpus, device_count);
        num_gpus = device_count;
    }

    if (seq_len % num_gpus != 0) {
        fprintf(stderr, "Sequence length %d must be divisible by num_gpus %d.\\n",
                seq_len, num_gpus);
        return EXIT_FAILURE;
    }

    FlashAttnConfig cfg;
    cfg.seq_len  = seq_len;
    cfg.dim      = dim;
    cfg.num_gpus = num_gpus;

    size_t total_elems = (size_t)seq_len * dim;
    size_t bytes = total_elems * sizeof(float);

    float* h_q   = (float*)malloc(bytes);
    float* h_k   = (float*)malloc(bytes);
    float* h_v   = (float*)malloc(bytes);
    float* h_out_single = (float*)malloc(bytes);
    float* h_out_multi  = (float*)malloc(bytes);

    if (!h_q || !h_k || !h_v || !h_out_single || !h_out_multi) {
        fprintf(stderr, "Host malloc failed.\\n");
        return EXIT_FAILURE;
    }

    srand(42);
    for (size_t i = 0; i < total_elems; ++i) {
        h_q[i] = rand_float();
        h_k[i] = rand_float();
        h_v[i] = rand_float();
    }

    // Single GPU baseline on device 0
    CHECK_CUDA(cudaSetDevice(0));
    float *d_q_full = NULL, *d_k_full = NULL, *d_v_full = NULL, *d_out_single = NULL;
    CHECK_CUDA(cudaMalloc(&d_q_full, bytes));
    CHECK_CUDA(cudaMalloc(&d_k_full, bytes));
    CHECK_CUDA(cudaMalloc(&d_v_full, bytes));
    CHECK_CUDA(cudaMalloc(&d_out_single, bytes));

    CHECK_CUDA(cudaMemcpy(d_q_full, h_q, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_full, h_k, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_full, h_v, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start_single, stop_single;
    CHECK_CUDA(cudaEventCreate(&start_single));
    CHECK_CUDA(cudaEventCreate(&stop_single));

    CHECK_CUDA(cudaEventRecord(start_single));
    flash_attn_single_gpu_forward(
        d_q_full, d_k_full, d_v_full, d_out_single, &cfg, 0);
    CHECK_CUDA(cudaEventRecord(stop_single));
    CHECK_CUDA(cudaEventSynchronize(stop_single));

    float ms_single = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_single, start_single, stop_single));
    printf("Single GPU forward: %.3f ms\\n", ms_single);

    CHECK_CUDA(cudaMemcpy(h_out_single, d_out_single, bytes, cudaMemcpyDeviceToHost));

    // Multi-GPU distributed version
    int local_seq_len = seq_len / num_gpus;
    size_t local_elems = (size_t)local_seq_len * dim;
    size_t local_bytes = local_elems * sizeof(float);

    float* d_q[8]   = {0};
    float* d_k[8]   = {0};
    float* d_v[8]   = {0};
    float* d_out[8] = {0};

    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_q[i], local_bytes));
        CHECK_CUDA(cudaMalloc(&d_k[i], local_bytes));
        CHECK_CUDA(cudaMalloc(&d_v[i], local_bytes));
        CHECK_CUDA(cudaMalloc(&d_out[i], local_bytes));

        size_t offset = (size_t)i * local_elems;
        CHECK_CUDA(cudaMemcpy(d_q[i], h_q + offset, local_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_k[i], h_k + offset, local_bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_v[i], h_v + offset, local_bytes, cudaMemcpyHostToDevice));
    }

    cudaEvent_t start_multi, stop_multi;
    CHECK_CUDA(cudaEventCreate(&start_multi));
    CHECK_CUDA(cudaEventCreate(&stop_multi));

    CHECK_CUDA(cudaSetDevice(0)); // timing on device 0 is fine
    CHECK_CUDA(cudaEventRecord(start_multi));

    flash_attn_multi_gpu_forward(d_q, d_k, d_v, d_out, &cfg);

    CHECK_CUDA(cudaEventRecord(stop_multi));
    CHECK_CUDA(cudaEventSynchronize(stop_multi));

    float ms_multi = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_multi, start_multi, stop_multi));
    printf("Multi-GPU forward: %.3f ms (%.2fx speedup vs single GPU)\\n",
           ms_multi, ms_single / ms_multi);

    // Gather outputs
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        size_t offset = (size_t)i * local_elems;
        CHECK_CUDA(cudaMemcpy(h_out_multi + offset, d_out[i],
                              local_bytes, cudaMemcpyDeviceToHost));
    }

    // Compare single-GPU and multi-GPU results
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < total_elems; ++i) {
        double diff = fabs((double)h_out_single[i] - (double)h_out_multi[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
        }
    }

    printf("Max |single - multi| difference: %.6e\\n", max_abs_diff);

    // Cleanup
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_q[i]));
        CHECK_CUDA(cudaFree(d_k[i]));
        CHECK_CUDA(cudaFree(d_v[i]));
        CHECK_CUDA(cudaFree(d_out[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d_q_full));
    CHECK_CUDA(cudaFree(d_k_full));
    CHECK_CUDA(cudaFree(d_v_full));
    CHECK_CUDA(cudaFree(d_out_single));

    free(h_q);
    free(h_k);
    free(h_v);
    free(h_out_single);
    free(h_out_multi);

    printf("Done.\\n");
    return 0;
}
