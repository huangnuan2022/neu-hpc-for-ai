
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CHECK_NCCL(call)                                                    \
    do {                                                                    \
        ncclResult_t _e = (call);                                           \
        if (_e != ncclSuccess) {                                            \
            fprintf(stderr, "NCCL error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, ncclGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

static inline float rand_float()
{
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; // [-1, 1]
}
