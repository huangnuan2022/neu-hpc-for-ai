# flash-attn2-c (CPU Reference)

FlashAttention-2 w tiling and the online normalizer.

```bash
make
```

## Run
```bash
./bin/flash_attn2_seq  [N d Br Bc seed]
# defaults: 512 64 128 128 42
```

## Files
- `src/flash_attn2_seq.c`
- `Makefile`
- `.gitignore`

License: MIT
