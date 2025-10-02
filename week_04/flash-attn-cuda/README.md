# FlashAttention HW4

## HW 4 Files
- `flash_attn_cuda.cu` — CUDA kernel + CPU oracle + driver.
- `Makefile` — tiny build script.
- `.gitignore` — ignores build artifacts.
- `demo.ipynb` — Colab-friendly notebook to compile & run

## Build & Run 
```bash
!nvcc -O3 -std=c++17 flash_attn_cuda.cu -o flash_attn_cuda
!./flash_attn_cuda 512 64 64 64 42
```

CLI: `./flash_attn_cuda N d BR BC seed`

## Additional details
- **Ownership:** grid only over row tiles (`BR` rows each). One block writes those rows of `O`.
- **KV loop inside block:** iterate all key/value tiles (`BC` columns) in shared memory.
- **Per-thread:** keep `q[i,:]` in registers; online `(m_t, l_t)` and tile contribution `T_row[:]` in registers from PMPP.
- **Merge:** after each KV tile, merge with running `(m_i, l_i, O[i,:])` using the online softmax formula from week3 lecture notes.

## License
MIT (educational/demo code).
