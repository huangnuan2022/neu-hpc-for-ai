CUDA GEMM Jupyter Notebook
Overview
cuda_gemm.ipynb is a Jupyter notebook designed for benchmarking general matrix-matrix multiplication (GEMM) on the GPU using custom CUDA C++ kernels. The notebook is runnable in Google Colab with GPU acceleration, and it demonstrates the development, compilation, and profiling of naive and tiled/shared-memory GEMM kernels.

Features
Checks and displays GPU availability and CUDA version.

Implements and benchmarks two custom CUDA GEMM kernels:

Naive GEMM (simple implementation).

Tiled shared-memory GEMM (optimized for speed).

Compiles CUDA code directly from notebook using nvcc.

Compares GPU results to a CPU-based reference for numerical correctness.

Measures and reports timing, throughput (GFLOPs), and GPU/CPU agreement.

Requirements
Google Colab or Jupyter with access to an NVIDIA GPU and CUDA toolkit.

CUDA 11+ and nvcc for compiling GPU kernels.

Python 3 kernel for running notebook cells and invoking system commands.

Usage
Open Notebook: Upload or open cuda_gemm.ipynb in Google Colab.

Set Runtime: Go to "Runtime" → "Change runtime type" and select "GPU".

Run All Cells: The notebook detects the GPU, writes the CUDA source code, compiles it, and runs GEMM benchmarks.

Inspect Results: Output cells display GEMM kernel timings, GFLOP/s, and error checks.

Example Output
text
Kernel naive M512 N512 K512 alpha1.25 beta0.75
Avg time 0.0002752 ms Throughput 977325 GFLOPs
Max GPU-CPU 43.7532

Kernel tiled M1024 N1024 K1024 alpha1 beta1
Avg time 0.000349867 ms Throughput 6.144e06 GFLOPs
Max GPU-CPU 51.3591
Customization
Change GEMM dimensions, alpha/beta, repetition count, and kernel type via command line arguments when running the benchmark binary.

Edit the gemm.cu cell to modify or add more kernels.

License
MIT or Apache 2.0, unless otherwise specified.

References
CUDA documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

Matrix multiplication/GEMM GPU optimization resources
