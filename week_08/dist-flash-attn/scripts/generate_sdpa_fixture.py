#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PyTorch SDPA FP32 correctness fixture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seq <= 0 or args.dim <= 0:
        raise SystemExit("seq and dim must be positive")

    torch.manual_seed(args.seed)
    q = (torch.rand(args.seq, args.dim, dtype=torch.float32) - 0.5) * 0.5
    k = (torch.rand(args.seq, args.dim, dtype=torch.float32) - 0.5) * 0.5
    v = (torch.rand(args.seq, args.dim, dtype=torch.float32) - 0.5) * 0.5
    reference = functional.scaled_dot_product_attention(
        q.view(1, 1, args.seq, args.dim),
        k.view(1, 1, args.seq, args.dim),
        v.view(1, 1, args.seq, args.dim),
        dropout_p=0.0,
        is_causal=False,
    ).view(args.seq, args.dim)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("reference", reference)):
        tensor.contiguous().numpy().tofile(args.output_dir / f"{name}.f32")
    (args.output_dir / "meta.txt").write_text(
        f"seq={args.seq}\ndim={args.dim}\nseed={args.seed}\nreference=pytorch_sdpa\n",
        encoding="utf-8",
    )
    print(f"Wrote PyTorch SDPA fixture to {args.output_dir}")


if __name__ == "__main__":
    main()
