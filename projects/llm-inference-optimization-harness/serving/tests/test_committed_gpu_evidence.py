from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_serving_system.benchmark_report import validate_artifact


EVIDENCE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "aws-a10g-1gpu-20260813"
    / "serving"
)


def test_committed_single_gpu_evidence_supports_documented_claims() -> None:
    artifact = json.loads((EVIDENCE_ROOT / "serving_benchmark.json").read_text(encoding="utf-8"))
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

    with (EVIDENCE_ROOT / "serving_requests.csv").open(newline="", encoding="utf-8") as handle:
        requests = list(csv.DictReader(handle))
    assert len(requests) == 285
    assert {row["status"] for row in requests} == {"ok"}
    assert {int(row["input_tokens"]) for row in requests} == {512}
    assert {int(row["requested_output_tokens"]) for row in requests} == {128}
    assert {int(row["output_tokens"]) for row in requests} == {128}
