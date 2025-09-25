GPU Programming – Week 3 README

This repo/notebook contains implementations and benchmarks for the Week-3 assignment:

In-place GEMM with transpose flags

HBM-aware tiled GEMM (shared memory)

Register-tiled GEMM (shared + micro-tiles)

Online normalizer softmax (Algorithm 3) from A Fast and Accurate Replacement for Softmax (arXiv:1805.02867)

All CUDA GEMM variants implement:

𝐶
←
𝛼
⋅
op
⁡
(
𝐴
)
op
⁡
(
𝐵
)
+
𝛽
⋅
𝐶
,
op
⁡
(
𝑋
)
∈
{
𝑋
,
  
𝑋
⊤
}
.
C←α⋅op(A)op(B)+β⋅C,op(X)∈{X,X
⊤
}.

No high-level CUDA libraries (no cuBLAS/cuDNN) are used.

Contents

gemm_xt.cu – In-place GEMM with optional transpose (tA, tB) + tiled shared-memory kernel and a naïve baseline.

gemm_xt_regtiling.cu – Register-tiled GEMM (shared tiles + per-thread micro-tiles).

softmax_online.c – Online normalizer softmax (Algorithm 3) + parity tests vs a safe two-pass reference.

Quick start (Colab)

Enable GPU → Runtime → Change runtime type → GPU.

Build

!nvcc -O3 -std=c++17 gemm_xt.cu -o gemm_xt
!nvcc -O3 -std=c++17 gemm_xt_regtiling.cu -o gemm_xt_regtiling
!gcc  -O3 -std=c11  softmax_online.c -lm -o softmax_online


Run (examples below).

GEMM: CLI & usage

Both CUDA binaries accept the same flags:

--m, --n, --k : output is m×n, reduction dim k

--tA N|T, --tB N|T : use op(A) = A or Aᵀ, op(B) = B or Bᵀ

--alpha <float>, --beta <float>

--repeat <int> : timing iterations (averaged)

--seed <int> : host RNG for inputs

For gemm_xt.cu only:

--kernel naive|tiled

--block 16|32 : shared tile size for the tiled kernel

For gemm_xt_regtiling.cu only:

--kernel reg|naive (register-tiling vs naïve)

Shapes (stored vs logical)

If tA=N: A stored as m×k; if tA=T: A stored as k×m.

If tB=N: B stored as k×n; if tB=T: B stored as n×k.

C is always stored m×n. (All arrays row-major.)

Sanity run (all transpose combos)
# Tiled (shared memory) — HBM-aware
!./gemm_xt --m 512 --n 384 --k 256 --tA N --tB N --alpha 1.2 --beta 0.7 --kernel tiled --block 32 --repeat 20
!./gemm_xt --m 512 --n 384 --k 256 --tA T --tB N --alpha 0.9 --beta 0.1 --kernel tiled --block 32 --repeat 20
!./gemm_xt --m 512 --n 384 --k 256 --tA N --tB T --alpha 1.0 --beta 0.0 --kernel tiled --block 32 --repeat 20
!./gemm_xt --m 512 --n 384 --k 256 --tA T --tB T --alpha 1.0 --beta 1.0 --kernel tiled --block 32 --repeat 20


You’ll see:

Max |C_gpu - C_ref| ≈ 1e-3–1e-4 for random inputs (float32).

Average time and GFLOP/s.

Naïve vs Tiled vs Register-Tiled
# Naïve baseline
!./gemm_xt --m 1024 --n 1024 --k 1024 --tA N --tB N --alpha 1 --beta 1 --kernel naive --repeat 10

# Tiled shared-memory kernel
!./gemm_xt --m 1024 --n 1024 --k 1024 --tA N --tB N --alpha 1 --beta 1 --kernel tiled --block 32 --repeat 20

# Register-tiling (shared + micro-tiles)
!./gemm_xt_regtiling --m 1024 --n 1024 --k 1024 --tA N --tB N --alpha 1 --beta 1 --kernel reg --repeat 20

What each CUDA kernel does
Naïve

Each thread computes one C[i,j].

Reads a full row of op(A) and column of op(B) from global memory → simple, bandwidth-hungry.

Tiled (shared memory)

Block computes a BLOCK×BLOCK output tile.

Cooperatively loads BLOCK×BLOCK tiles of op(A) and op(B) into shared memory, reusing each value BLOCK times.

--block {16,32}; +1 padding in shared arrays reduces bank conflicts.

Fused epilogue: in-place C[i,j] = α·acc + β·C[i,j].

Register-tiled (shared + micro-tiles)

Block computes a large tile (e.g., 64×64), traversing K in chunks (e.g., KT=16).

Each thread accumulates a TM×TN patch (default 2×2) in registers.

Higher arithmetic intensity and fewer global loads than the plain tiled kernel.

Default config in gemm_xt_regtiling.cu:

Block tile BLOCK_M×BLOCK_N = 64×64

K tile KT = 16

Micro-tile TM×TN = 2×2

Threads per block = (BLOCK_N/TN) × (BLOCK_M/TM) = 32 × 32 = 1024

Want to explore? Recompile with different template constants (e.g., 128×64×8, TM×TN=4×2) and measure.

Online softmax (Algorithm 3)

Implements the single-pass, numerically stable softmax with on-the-fly max and normalizer updates.

Build & test

!gcc -O3 -std=c11 softmax_online.c -lm -o softmax_online
!./softmax_online 4096 100


Expected output

Very small max_abs_diff and avg_abs_diff vs a safe two-pass reference.

sum(y) ≈ 1.0 (within ~1e-6–1e-7).

Algorithm (pseudocode)

m = -inf
d = 0
for xj in x:
    mj = max(m, xj)
    d  = d * exp(m - mj) + exp(xj - mj)
    m  = mj
y_i = exp(x_i - m) / d


Reference: A Fast and Accurate Replacement for Softmax (arXiv:1805.02867), Algorithm 3.

Design notes & pitfalls

In-place semantics: We always read the original C and write back α·acc + β·C. No extra D buffer is allocated.

Transpose handling: Done at global→shared load time via helper indexers; ensures coalesced reads for both N and T.

Numerical accuracy: Float32 accumulations; differences vs CPU double-accumulation are expected at ~1e-3 for large random matrices.

Block sizes & occupancy:

Tiled: --block 32 often best; try 16 on smaller GPUs.

Register-tiling: 64×64 with 1024 threads is a good, safe default—benchmark on your GPU.

Bank conflicts: Shared tiles are padded (+1) in one dimension.

Repro tips

Use larger sizes (e.g., 2048+) to approach steady-state bandwidth and higher GFLOP/s.

Increase --repeat for stable timing.

Fix the RNG --seed to compare kernels apples-to-apples.
