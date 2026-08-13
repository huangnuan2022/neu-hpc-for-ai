from __future__ import annotations

from copy import deepcopy

import pytest

from llm_serving_system.regression_gate import compare_artifacts


def artifact() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-08-13T00:00:00+00:00",
        "performance_evidence_valid": True,
        "config": {
            "model": "Qwen/Qwen3-8B",
            "deployment": "aws-a10g-1gpu",
            "backend_kind": "vllm",
            "tokenizer_mode": "huggingface",
            "worker_count": 1,
            "input_tokens": 512,
            "output_tokens": 128,
            "concurrency": [1],
            "warmup_runs": 1,
            "measured_runs": 5,
        },
        "environment": {
            "git_sha": "abc",
            "git_dirty": False,
            "gpus": [{"index": 0}],
            "vllm_image": {"image_id": "sha256:abc"},
        },
        "summaries": [
            {
                "worker_count": 1,
                "concurrency": 1,
                "mode": "sequential",
                "request_count": 5,
                "aggregate_output_tokens_per_second": 100.0,
                "requests_per_second": 1.0,
                "error_rate": 0.0,
                "ttft_seconds": {"p50": 0.8, "p95": 1.0, "p99": 1.1},
                "tpot_seconds": {"p50": 0.02, "p95": 0.03, "p99": 0.04},
                "e2e_seconds": {"p50": 2.0, "p95": 3.0, "p99": 3.2},
            }
        ],
        "runs": [],
    }


def test_regression_gate_accepts_values_inside_budget() -> None:
    baseline = artifact()
    current = deepcopy(baseline)
    current["summaries"][0]["aggregate_output_tokens_per_second"] = 91.0
    current["summaries"][0]["ttft_seconds"]["p95"] = 1.09
    assert compare_artifacts(baseline, current, 0.10) == []


def test_regression_gate_reports_throughput_and_latency_failures() -> None:
    baseline = artifact()
    current = deepcopy(baseline)
    current["summaries"][0]["aggregate_output_tokens_per_second"] = 89.0
    current["summaries"][0]["ttft_seconds"]["p95"] = 1.11
    failures = compare_artifacts(baseline, current, 0.10)
    assert any("throughput" in failure for failure in failures)
    assert any("ttft" in failure for failure in failures)


def test_regression_gate_rejects_fake_artifacts() -> None:
    baseline = artifact()
    current = deepcopy(baseline)
    current["performance_evidence_valid"] = False
    with pytest.raises(ValueError, match="real-GPU"):
        compare_artifacts(baseline, current, 0.10)
