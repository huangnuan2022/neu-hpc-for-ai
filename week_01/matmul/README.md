# Matrix Multiplication (Single-threaded & Multi-threaded)

This project implements matrix multiplication in C, both single-threaded and multi-threaded (using `pthreads`).  
It was built to satisfy the assignment requirements:

- Implement a single-threaded version in C.
- Implement a multi-threaded version in C using pthreads.
- Write test cases that cover a wide range of dimensions, including corner cases.
- Verify that the multi-threaded version produces identical results to the single-threaded version.
- Benchmark and measure speedup for different thread counts: **1, 4, 16, 32, 64, 128**.
- Use large matrices so that speedup is measurable.

---

## Build & Run

### Requirements

- GCC or Clang with pthread support
- POSIX system (Linux, macOS, WSL)

### Build

make

### Run

./matmul

### Clean

make clean

```

```
