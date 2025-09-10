#ifndef MATMUL_H
#define MATMUL_H

#include "mat.h"
#include <stddef.h>

// Multiply C = A x B (serial, single-threaded).
// Requirements: A.cols == B.rows, C is (A.rows x B.cols).
// C will be fully overwritten (not accumulated).
void matmul_serial(const Mat* A, const Mat* B, Mat* C);

// Multiply C = A x B using pthreads (row-blocking).
// num_threads > 0. If num_threads > A.rows, it will be clamped to A.rows.
void matmul_threads(const Mat* A, const Mat* B, Mat* C, size_t num_threads);

#endif // MATMUL_H
