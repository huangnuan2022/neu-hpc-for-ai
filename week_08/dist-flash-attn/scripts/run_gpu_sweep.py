#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"benchmark emitted no JSON: {result.stdout}\n{result.stderr}")


def markdown(artifact: dict[str, Any]) -> str:
    lines = [
        "# CUDA/NCCL Attention Benchmark",
        "",
        f"- Generated at: `{artifact['generated_at']}`",
        f"- Git SHA: `{artifact['environment']['git_sha']}`",
        f"- GPU inventory: `{artifact['environment']['nvidia_smi']}`",
        f"- CUDA compiler: `{artifact['environment']['nvcc']}`",
        f"- Warm-up / iterations: `{artifact['warmup']}` / `{artifact['iterations']}`",
        "",
        "## Correctness Against PyTorch SDPA",
        "",
        "| GPUs | max abs error | max rel error | reference |",
        "| ---: | ---: | ---: | --- |",
    ]
    for result in artifact["correctness"]:
        lines.append(
            f"| {result['gpus']} | {result['ring_max_abs_error']:.3e} | "
            f"{result['ring_max_rel_error']:.3e} | {result['reference']} |"
        )
    lines.extend(
        [
            "",
            "## Steady-State Performance",
            "",
            "| seq | GPUs | overlap | median | p95 | speedup vs 1 GPU | minimal state reduction | explicit workspace reduction |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in artifact["performance"]:
        lines.append(
            f"| {result['seq']} | {result['gpus']} | {result['overlap_kv_rotation']} | "
            f"{result['ring_median_ms']:.3f} ms | {result['ring_p95_ms']:.3f} ms | "
            f"{result['speedup_vs_single_gpu']:.2f}x | "
            f"{result['estimated_minimal_state_reduction_pct']:.2f}% | "
            f"{result['explicit_workspace_reduction_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- CUDA/NCCL communicators, streams, events, and buffers are created before warm-up.",
            "- Each sample uses CUDA events and reports the maximum elapsed duration across participating devices.",
            "- K/V input staging is outside timed forward execution; state initialization, attention kernels, NCCL rotation, and output finalization are timed.",
            "- The overlap comparison uses separate compute/communication streams and double-buffered K/V rotation.",
            "- Minimal state is formula-based; explicit workspace is the exact sum of cudaMalloc requests; measured allocation deltas are retained in JSON.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 1/2/4-GPU attention benchmark matrix")
    parser.add_argument("--binary", type=Path, default=ROOT / "build" / "dist_flash_attn")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gpus", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--correctness-case", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 5 or args.warmup < 1:
        raise SystemExit("require at least one warm-up and five measured iterations")

    gpu_counts = [value for value in (1, 2, 4) if value <= args.max_gpus]
    common = ["--warmup", str(args.warmup), "--iterations", str(args.iterations)]
    correctness = [
        run_json(
            [
                str(args.binary),
                "--case-dir",
                str(args.correctness_case),
                "--gpus",
                str(gpus),
                *common,
            ]
        )
        for gpus in gpu_counts
    ]
    performance: list[dict[str, Any]] = []
    for seq in (1024, 2048, 4096):
        for gpus in gpu_counts:
            base = [
                str(args.binary),
                "--seq",
                str(seq),
                "--dim",
                "64",
                "--gpus",
                str(gpus),
                *common,
            ]
            performance.append(run_json(base))
            if gpus > 1:
                performance.append(run_json([*base, "--no-overlap"]))

    artifact = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "environment": {
            "platform": platform.platform(),
            "git_sha": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "nvidia_smi": command_output(["nvidia-smi", "-L"]),
            "nvcc": command_output(["nvcc", "--version"]),
        },
        "correctness": correctness,
        "performance": performance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "cuda_attention_benchmark.json"
    markdown_path = args.output_dir / "cuda_attention_benchmark.md"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown(artifact), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
