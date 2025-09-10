#include "matmul.h"
#include <string.h>
#include <stdio.h>    // fprintf, stderr
#include <stdlib.h>   // exit


// C = A x B (serial)
// Row-major, cache-friendly loop order: i -> k -> j
void matmul_serial(const Mat* A, const Mat* B, Mat* C) {
    // Basic shape checks (defensive)
    if (A->cols != B->rows || C->rows != A->rows || C->cols != B->cols) {
        fprintf(stderr, "Dimension mismatch in matmul_serial.\n");
        exit(1);
    }

    const size_t M = A->rows, K = A->cols, N = B->cols;
    const float* a = A->data;
    const float* b = B->data;
    float* c = C->data;

    // Overwrite C
    memset(c, 0, sizeof(float) * M * N);

    for (size_t i = 0; i < M; ++i) {
        const size_t ci = i * N;
        const size_t ai = i * K;
        for (size_t k = 0; k < K; ++k) {
            const float aik = a[ai + k];      // Keep A(i,k) in a register
            const size_t bk = k * N;          // B row offset
            for (size_t j = 0; j < N; ++j) {  // Walk contiguous row segments of B and C
                c[ci + j] += aik * b[bk + j];
            }
        }
    }
}
