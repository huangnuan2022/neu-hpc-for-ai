#include "mat.h"
#include <stdlib.h>
#include <string.h>

Mat mat_alloc(size_t rows, size_t cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    size_t bytes = rows * cols * sizeof(float);
    m.data = (bytes ? (float*)malloc(bytes) : NULL);
    return m;
}

Mat mat_alloc_zero(size_t rows, size_t cols) {
    Mat m = mat_alloc(rows, cols);
    if (m.data) {
        size_t bytes = rows * cols * sizeof(float);
        memset(m.data, 0, bytes);
    }
    return m;
}

void mat_free(Mat* m) {
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->rows = m->cols = 0;
}
