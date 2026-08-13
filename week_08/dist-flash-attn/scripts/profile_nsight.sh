#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 BINARY OUTPUT_PREFIX [GPU_COUNT]" >&2
  exit 2
fi

binary="$1"
output_prefix="$2"
gpu_count="${3:-4}"

command -v nsys >/dev/null 2>&1 || {
  echo "nsys is required" >&2
  exit 1
}

mkdir -p "$(dirname "$output_prefix")"
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output="$output_prefix" \
  "$binary" --seq 4096 --dim 64 --gpus "$gpu_count" --warmup 2 --iterations 5

nsys stats --report cuda_gpu_kern_sum,cuda_api_sum "$output_prefix.nsys-rep" \
  > "$output_prefix.stats.txt"
