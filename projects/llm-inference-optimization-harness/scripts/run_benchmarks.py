#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "inference_harness"
RESULTS = ROOT / "results"


ATTENTION_CONFIGS = [
    ["attention", "--seq", "128", "--dim", "32", "--shards", "2", "--iters", "3"],
    ["attention", "--seq", "256", "--dim", "64", "--shards", "4", "--iters", "3"],
    ["attention", "--seq", "512", "--dim", "64", "--shards", "4", "--iters", "2"],
    ["attention", "--seq", "1024", "--dim", "64", "--shards", "4", "--iters", "1"],
]

MOE_CONFIGS = [
    ["moe", "--tokens", "128", "--dim", "32", "--hidden", "64", "--experts", "8", "--shards", "4", "--iters", "3"],
    ["moe", "--tokens", "256", "--dim", "32", "--hidden", "64", "--experts", "8", "--shards", "4", "--iters", "3"],
    ["moe", "--tokens", "512", "--dim", "32", "--hidden", "64", "--experts", "8", "--shards", "4", "--iters", "2"],
]


def run_case(args: list[str]) -> dict:
    proc = subprocess.run(
        [str(BIN), *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    row = json.loads(proc.stdout)
    if row["mode"] == "attention":
        max_diff = max(row["max_diff_streaming"], row["max_diff_ring"])
        if max_diff > 1e-5:
            raise RuntimeError(f"attention correctness gate failed: max_diff={max_diff}")
    elif row["mode"] == "moe":
        if row["max_diff"] > 1e-5:
            raise RuntimeError(f"moe correctness gate failed: max_diff={row['max_diff']}")
    return row


def write_csv(rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row.keys()})
    with (RESULTS / "benchmark_results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_ms(value: float) -> str:
    return f"{value:.3f} ms"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def write_markdown(rows: list[dict], metadata: dict) -> None:
    attention_rows = [row for row in rows if row["mode"] == "attention"]
    moe_rows = [row for row in rows if row["mode"] == "moe"]

    max_attention_diff = max(max(row["max_diff_streaming"], row["max_diff_ring"]) for row in attention_rows)
    max_moe_diff = max(row["max_diff"] for row in moe_rows)
    best_memory_reduction = max(row["estimated_memory_reduction_pct"] for row in attention_rows)
    largest_attention = max(attention_rows, key=lambda row: row["seq"])
    largest_moe = max(moe_rows, key=lambda row: row["tokens"])

    lines = [
        "# Benchmark Summary",
        "",
        f"- Generated at: `{metadata['generated_at']}`",
        f"- Host: `{metadata['platform']}`",
        "- Compiler binary: `build/inference_harness`",
        "",
        "## Key Results",
        "",
        f"- Verified streaming and ring-sequence-parallel attention against a materialized softmax reference across `{len(attention_rows)}` shape sweeps.",
        f"- Verified routed expert-sharded MoE execution against a dense top-1 reference across `{len(moe_rows)}` shape sweeps.",
        f"- Maximum attention correctness drift: `{fmt_sci(max_attention_diff)}`.",
        f"- Maximum MoE correctness drift: `{fmt_sci(max_moe_diff)}`.",
        f"- Largest attention sweep: `seq={largest_attention['seq']}`, `dim={largest_attention['dim']}`, `shards={largest_attention['shards']}`.",
        f"- Estimated peak per-shard attention working-memory reduction versus materializing the score matrix: `{best_memory_reduction:.2f}%`.",
        "",
        "## Attention Sweeps",
        "",
        "| seq | dim | shards | naive | streaming | ring sim | max ring diff | memory reduction |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in attention_rows:
        lines.append(
            "| {seq} | {dim} | {shards} | {naive} | {streaming} | {ring} | {diff} | {mem:.2f}% |".format(
                seq=row["seq"],
                dim=row["dim"],
                shards=row["shards"],
                naive=fmt_ms(row["naive_ms"]),
                streaming=fmt_ms(row["streaming_ms"]),
                ring=fmt_ms(row["ring_ms"]),
                diff=fmt_sci(row["max_diff_ring"]),
                mem=row["estimated_memory_reduction_pct"],
            )
        )

    lines.extend(
        [
            "",
            "## MoE Routing Sweeps",
            "",
            "| tokens | dim | hidden | experts | shards | dense | routed sim | max diff | route imbalance |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in moe_rows:
        lines.append(
            "| {tokens} | {dim} | {hidden} | {experts} | {shards} | {dense} | {routed} | {diff} | {imbalance:.2f}x |".format(
                tokens=row["tokens"],
                dim=row["dim"],
                hidden=row["hidden"],
                experts=row["experts"],
                shards=row["shards"],
                dense=fmt_ms(row["dense_ms"]),
                routed=fmt_ms(row["routed_ms"]),
                diff=fmt_sci(row["max_diff"]),
                imbalance=row["route_imbalance"],
            )
        )

    lines.extend(
        [
            "",
            "## Resume-Safe Claims",
            "",
            f"- Correctness-gated `{len(rows)}` benchmark sweeps for attention and MoE inference primitives.",
            f"- Attention ring simulator preserved output parity within `{fmt_sci(max_attention_diff)}` max absolute error.",
            f"- Routed MoE simulator preserved output parity within `{fmt_sci(max_moe_diff)}` max absolute error on the tested shapes.",
            f"- Largest measured attention shape avoided materializing a `{largest_attention['seq']} x {largest_attention['seq']}` score matrix in the streaming/ring path.",
            f"- Largest MoE sweep routed `{largest_moe['tokens']}` tokens across `{largest_moe['experts']}` experts and `{largest_moe['shards']}` logical shards.",
            "",
        ]
    )

    (RESULTS / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not BIN.exists():
        raise SystemExit(f"missing binary: {BIN}; run `make` first")

    RESULTS.mkdir(exist_ok=True)
    rows = [run_case(args) for args in ATTENTION_CONFIGS + MOE_CONFIGS]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cases": len(rows),
    }
    payload = {"metadata": metadata, "results": rows}

    (RESULTS / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(rows)
    write_markdown(rows, metadata)

    print(f"wrote {RESULTS / 'benchmark_results.json'}")
    print(f"wrote {RESULTS / 'benchmark_results.csv'}")
    print(f"wrote {RESULTS / 'benchmark_summary.md'}")


if __name__ == "__main__":
    main()
