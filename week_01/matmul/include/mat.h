#ifndef MAT_H
#define MAT_H

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* data;  // Row-major: element(i,j) at data[i*cols + j]
} Mat;

// Allocate (rows x cols) matrix with uninitialized data.
// Returns NULL on failure.
Mat mat_alloc(size_t rows, size_t cols);

// Allocate (rows x cols) and zero it.
Mat mat_alloc_zero(size_t rows, size_t cols);

// Free matrix memory (safe on empty data).
void mat_free(Mat* m);

// Element access helpers (no bounds checking).
static inline float  mat_get(const Mat* m, size_t i, size_t j) {
    return m->data[i * m->cols + j];
}
static inline void   mat_set(Mat* m, size_t i, size_t j, float v) {
    m->data[i * m->cols + j] = v;
}

#endif // MAT_H
