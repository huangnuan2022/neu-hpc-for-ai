#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>
#include <stdbool.h>

// Safe malloc that exits on OOM with a message.
void* xmalloc(size_t bytes);

// Monotonic time in seconds for benchmarking.
double now_sec(void);

// Fill buffer with sequential values: base + step*i
void fill_sequential(float* a, size_t n, float base, float step);

// Fill buffer with pseudo-random values in [-1, 1] with fixed seed for reproducibility.
void fill_random(float* a, size_t n, unsigned seed);

// Compare two float arrays with combined relative/absolute tolerance.
bool array_almost_equal(const float* a, const float* b, size_t n,
                        float rel_eps, float abs_eps);

#endif // UTIL_H
