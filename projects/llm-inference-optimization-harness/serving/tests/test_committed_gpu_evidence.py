from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_serving_system.benchmark_report import validate_artifact


SINGLE_GPU_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "aws-a10g-1gpu-20260813"
)
FOUR_GPU_ROOT = Path(__file__).resolve().parents[2] / "results" / "aws-a10g-4gpu-20260821"


def test_committed_single_gpu_evidence_supports_documented_claims() -> None:
    evidence_root = SINGLE_GPU_ROOT / "serving"
    artifact = json.loads((evidence_root / "serving_benchmark.json").read_text(encoding="utf-8"))
    validate_artifact(artifact)
    assert artifact["performance_evidence_valid"] is True
    assert artifact["environment"]["git_sha"] == "c5cbb271aa8cbe738ac74a69a74d53ee40ac7040"

    summaries = {summary["concurrency"]: summary for summary in artifact["summaries"]}
    concurrency_32 = summaries[32]
    assert concurrency_32["aggregate_output_tokens_per_second"] == pytest.approx(739.8683636337271)
    assert concurrency_32["ttft_seconds"]["p95"] == pytest.approx(0.35220835894997443)
    assert concurrency_32["e2e_seconds"]["p95"] == pytest.approx(5.537492897700031)
    assert concurrency_32["throughput_vs_sequential"] == pytest.approx(25.369675209465164)
    assert concurrency_32["estimated_cost_per_million_output_tokens_usd"] == pytest.approx(
        0.3776948146181094
    )

    with (evidence_root / "serving_requests.csv").open(newline="", encoding="utf-8") as handle:
        requests = list(csv.DictReader(handle))
    assert len(requests) == 285
    assert {row["status"] for row in requests} == {"ok"}
    assert {int(row["input_tokens"]) for row in requests} == {512}
    assert {int(row["requested_output_tokens"]) for row in requests} == {128}
    assert {int(row["output_tokens"]) for row in requests} == {128}


def test_committed_four_gpu_serving_evidence_supports_documented_claims() -> None:
    evidence_root = FOUR_GPU_ROOT / "serving"
    artifact = json.loads((evidence_root / "serving_benchmark.json").read_text(encoding="utf-8"))
    validate_artifact(artifact)
    assert artifact["performance_evidence_valid"] is True
    assert artifact["config"]["deployment"] == "aws-a10g-4gpu"
    assert artifact["config"]["worker_count"] == 4
    assert artifact["config"]["model"] == "Qwen/Qwen3-8B"
    assert artifact["environment"]["git_sha"] == "e056e29e326868f7a88405f6ae13b3a02f2d3bf7"
    assert len(artifact["environment"]["gpus"]) == 4
    assert {gpu["name"] for gpu in artifact["environment"]["gpus"]} == {"NVIDIA A10G"}

    summaries = {summary["concurrency"]: summary for summary in artifact["summaries"]}
    concurrency_32 = summaries[32]
    assert concurrency_32["aggregate_output_tokens_per_second"] == pytest.approx(
        786.4079403158808
    )
    assert concurrency_32["ttft_seconds"]["p95"] == pytest.approx(0.24305352150003046)
    assert concurrency_32["e2e_seconds"]["p95"] == pytest.approx(5.195045289099959)
    assert concurrency_32["estimated_cost_per_million_output_tokens_usd"] == pytest.approx(
        2.0034837833945236
    )
    assert concurrency_32["failed_requests"] == 0
    assert concurrency_32["timeout_requests"] == 0

    with (evidence_root / "serving_requests.csv").open(newline="", encoding="utf-8") as handle:
        requests = list(csv.DictReader(handle))
    assert len(requests) == 285
    assert {row["status"] for row in requests} == {"ok"}
    assert {int(row["input_tokens"]) for row in requests} == {512}
    assert {int(row["requested_output_tokens"]) for row in requests} == {128}
    assert {int(row["output_tokens"]) for row in requests} == {128}
    assert {row["backend"] for row in requests} == {
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-3",
    }


def test_committed_four_gpu_cuda_evidence_supports_documented_claims() -> None:
    artifact = json.loads(
        (FOUR_GPU_ROOT / "cuda-attention" / "cuda_attention_benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["warmup"] == 5
    assert artifact["iterations"] == 20
    assert {row["gpus"] for row in artifact["correctness"]} == {1, 2, 4}
    assert max(row["ring_max_abs_error"] for row in artifact["correctness"]) == pytest.approx(
        2.23517418e-08
    )

    four_gpu = next(
        row
        for row in artifact["performance"]
        if row["seq"] == 4096 and row["gpus"] == 4 and row["overlap_kv_rotation"]
    )
    serialized = next(
        row
        for row in artifact["performance"]
        if row["seq"] == 4096 and row["gpus"] == 4 and not row["overlap_kv_rotation"]
    )
    assert four_gpu["ring_median_ms"] == pytest.approx(7.37996793)
    assert four_gpu["speedup_vs_single_gpu"] == pytest.approx(2.486194)
    assert four_gpu["estimated_minimal_state_reduction_pct"] == pytest.approx(98.815918)
    assert four_gpu["explicit_workspace_reduction_pct"] == pytest.approx(98.034668)
    overlap_latency_reduction = 1.0 - four_gpu["ring_median_ms"] / serialized["ring_median_ms"]
    assert overlap_latency_reduction == pytest.approx(0.0648760682913939)

    nsight_report = FOUR_GPU_ROOT / "nsight" / "attention-seq4096-4gpu.nsys-rep"
    assert nsight_report.stat().st_size > 500_000
