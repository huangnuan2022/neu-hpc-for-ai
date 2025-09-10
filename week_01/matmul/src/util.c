#define _POSIX_C_SOURCE 200809L
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void* xmalloc(size_t bytes) {
    void* p = malloc(bytes);
    if (!p && bytes) {
        fprintf(stderr, "Out of memory: requested %zu bytes\n", bytes);
        exit(1);
    }
    return p;
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void fill_sequential(float* a, size_t n, float base, float step) {
    for (size_t i = 0; i < n; ++i) a[i] = base + step * (float)i;
}

void fill_random(float* a, size_t n, unsigned seed) {
    // Fixed seed → deterministic tests
    srand(seed);
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)((rand() / (double)RAND_MAX) * 2.0 - 1.0);
    }
}

bool array_almost_equal(const float* a, const float* b, size_t n,
                        float rel_eps, float abs_eps) {
    for (size_t i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        float scale = fmaxf(fabsf(a[i]), fabsf(b[i]));
        float tol = fmaxf(abs_eps, rel_eps * scale);
        if (diff > tol) {
            fprintf(stderr, "Mismatch at %zu: %g vs %g (tol=%g)\n", i, a[i], b[i], tol);
            return false;
        }
    }
    return true;
}
