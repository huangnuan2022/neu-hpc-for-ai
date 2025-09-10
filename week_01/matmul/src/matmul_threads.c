#include "matmul.h"
#include "util.h"
#include <pthread.h>
#include <string.h>
#include <stdio.h>    // fprintf
#include <stdlib.h>   // exit, free


typedef struct {
    const float* A;
    const float* B;
    float*       C;
    size_t M, K, N;
    size_t row_start; // inclusive
    size_t row_end;   // exclusive
} Task;

// Worker computes rows [row_start, row_end) of C.
// We keep the same i->k->j order inside each thread for cache locality.
static void* worker(void* arg) {
    const Task* t = (const Task*)arg;
    const float* a = t->A;
    const float* b = t->B;
    float*       c = t->C;

    for (size_t i = t->row_start; i < t->row_end; ++i) {
        const size_t ci = i * t->N;
        const size_t ai = i * t->K;
        memset(c + ci, 0, sizeof(float) * t->N);

        for (size_t k = 0; k < t->K; ++k) {
            const float aik = a[ai + k];
            const size_t bk = k * t->N;
            for (size_t j = 0; j < t->N; ++j) {
                c[ci + j] += aik * b[bk + j];
            }
        }
    }
    return NULL;
}

void matmul_threads(const Mat* A, const Mat* B, Mat* C, size_t num_threads) {
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Dimension mismatch in matmul_threads.\n");
        exit(1);
    }
    if (num_threads == 0) {
        fprintf(stderr, "num_threads must be > 0\n");
        exit(1);
    }

    const size_t M = A->rows, K = A->cols, N = B->cols;
    if (num_threads > M) num_threads = M; // No more threads than rows

    pthread_t* th = (pthread_t*)xmalloc(sizeof(pthread_t) * num_threads);
    Task* tasks   = (Task*)xmalloc(sizeof(Task) * num_threads);

    // Row-blocking with ceiling chunk
    size_t chunk = (M + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk;
        size_t end   = start + chunk;
        if (start > M) start = M;
        if (end   > M) end   = M;

        tasks[t] = (Task){
            .A = A->data, .B = B->data, .C = C->data,
            .M = M, .K = K, .N = N,
            .row_start = start,
            .row_end   = end
        };
        pthread_create(&th[t], NULL, worker, &tasks[t]);
    }

    for (size_t t = 0; t < num_threads; ++t) {
        pthread_join(th[t], NULL);
    }

    free(tasks);
    free(th);
}
