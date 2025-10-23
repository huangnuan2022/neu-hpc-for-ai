# flashattn-v2-ring-c 
## Build

```bash
make            
# CC=gcc CFLAGS="-O3 -march=native" make
```

## Run



```bash
#               S    d   N  seed
./fa2_ring      1024 64  4  42
./fa2_ring      2048 128 8  1
```

### Arguments

```
usage: ./fa2_ring [S d N seed]
  S     total sequence length (must be divisible by N; default 1024)
  d     head dimension (default 64)
  N     logical devices (1..8; default 4)
```

### Notes



## License

MIT
