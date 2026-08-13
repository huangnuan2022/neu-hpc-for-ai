from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from llm_serving_system.benchmark_environment import parse_vllm_metrics
from llm_serving_system.attention_memory import attention_memory
from llm_serving_system.benchmark_metrics import add_sequential_comparisons, percentile, summarize_scenario
from llm_serving_system.benchmark_report import validate_artifact, write_artifacts
from llm_serving_system.fake_backend import create_app as create_fake_app
from llm_serving_system.load_benchmark import (
    SSEParser,
    WhitespaceTokenizer,
    WorkloadConfig,
    execute_request,
)


def test_sse_parser_handles_chunk_boundaries_usage_and_done() -> None:
    parser = SSEParser()
    first = b'data: {"choices":[{"text":"one "}]}'
    second = b'\n\ndata: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n'
    assert parser.feed(first) == []
    events = parser.feed(second)
    assert events[0]["choices"][0]["text"] == "one "
    assert events[1]["usage"]["completion_tokens"] == 1
    assert events[2] == "[DONE]"


def test_execute_request_records_streaming_metrics() -> None:
    async def scenario() -> None:
        transport = httpx.ASGITransport(app=create_fake_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            record = await execute_request(
                client,
                WhitespaceTokenizer(),
                WorkloadConfig(
                    endpoint="http://test",
                    deployment="test",
                    worker_count=1,
                    input_tokens=32,
                    output_tokens=3,
                    prefix_groups=2,
                    timeout_seconds=2.0,
                ),
                concurrency=1,
                run_index=0,
                request_index=0,
            )
        assert record["status"] == "ok"
        assert record["input_tokens"] == 32
        assert record["output_tokens"] == 3
        assert record["ttft_seconds"] is not None
        assert record["tpot_seconds"] is not None
        assert record["e2e_seconds"] >= record["ttft_seconds"]

    asyncio.run(scenario())


def test_percentiles_summary_and_sequential_comparison() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    records = [
        {
            "status": "ok",
            "output_tokens": 10,
            "backend": "worker-0",
            "route_reason": "prefix_affinity",
            "queue_depth_at_route": 0,
            "ttft_seconds": 1.0,
            "tpot_seconds": 0.1,
            "inter_chunk_seconds": [0.1, 0.2],
            "e2e_seconds": 2.0,
        },
        {
            "status": "timeout",
            "output_tokens": 0,
            "backend": None,
            "route_reason": None,
            "queue_depth_at_route": 0,
            "ttft_seconds": None,
            "tpot_seconds": None,
            "inter_chunk_seconds": [],
            "e2e_seconds": 3.0,
        },
    ]
    sequential = summarize_scenario(records, measured_wall_seconds=4.0, hourly_cost_usd=4.0)
    sequential.update({"concurrency": 1})
    concurrent = dict(sequential, concurrency=8, aggregate_output_tokens_per_second=5.0)
    add_sequential_comparisons([sequential, concurrent])
    assert sequential["error_rate"] == 0.5
    assert sequential["aggregate_output_tokens_per_second"] == 2.5
    assert sequential["estimated_cost_per_million_output_tokens_usd"] == 444.44444444444446
    assert concurrent["throughput_vs_sequential"] == 2.0


def test_vllm_prometheus_metric_parser_tracks_cache_and_queue() -> None:
    text = """
# HELP ignored ignored
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-8B"} 0.75
vllm:num_requests_running{model_name="Qwen/Qwen3-8B"} 3
vllm:num_requests_waiting{model_name="Qwen/Qwen3-8B"} 2
unrelated_metric 99
"""
    values = parse_vllm_metrics(text)
    assert values["vllm:gpu_cache_usage_perc"] == 0.75
    assert values["vllm:num_requests_running"] == 3.0
    assert "unrelated_metric" not in values


def test_report_schema_and_three_artifact_formats(tmp_path: Path) -> None:
    summary = {
        "deployment": "test",
        "worker_count": 1,
        "concurrency": 1,
        "mode": "sequential",
        "request_count": 1,
        "aggregate_output_tokens_per_second": 1.0,
        "requests_per_second": 1.0,
        "error_rate": 0.0,
        "ttft_seconds": {"p50": 0.1, "p95": 0.1, "p99": 0.1},
        "tpot_seconds": {"p50": 0.01, "p95": 0.01, "p99": 0.01},
        "e2e_seconds": {"p50": 0.2, "p95": 0.2, "p99": 0.2},
        "backend_counts": {"worker-0": 1},
        "route_reason_counts": {"prefix_affinity": 1},
        "affinity_route_rate": 1.0,
        "throughput_vs_sequential": 1.0,
        "estimated_cost_per_million_output_tokens_usd": None,
    }
    artifact = {
        "schema_version": "1.0",
        "generated_at": "2026-08-13T00:00:00+00:00",
        "performance_evidence_valid": False,
        "config": {
            "model": "Qwen/Qwen3-8B",
            "deployment": "test",
            "backend_kind": "fake",
            "worker_count": 1,
            "input_tokens": 512,
            "output_tokens": 128,
            "concurrency": [1],
            "warmup_runs": 1,
            "measured_runs": 5,
        },
        "environment": {"git_sha": "abc", "git_dirty": False, "aws": {}, "gpus": []},
        "summaries": [summary],
        "runs": [{"requests": [{"deployment": "test"}]}],
        "gpu_summary": {},
        "vllm_metrics_summary": {},
    }
    validate_artifact(artifact)
    paths = write_artifacts(artifact, tmp_path)
    assert set(paths) == {"json", "csv", "markdown"}
    assert all(path.exists() for path in paths.values())
    assert "not performance evidence" in paths["markdown"].read_text(encoding="utf-8")


def test_4096_four_gpu_attention_memory_formula() -> None:
    memory = attention_memory(seq=4096, dim=64, gpus=4)
    assert memory["full_score_matrix_bytes"] == 67_108_864
    assert memory["minimal_ring_state_bytes_per_gpu"] == 794_624
    assert memory["explicit_double_buffer_workspace_bytes_per_gpu"] == 1_318_912
    assert abs(memory["minimal_state_reduction_pct"] - 98.81591796875) < 1e-12
    assert abs(memory["explicit_workspace_reduction_pct"] - 98.03466796875) < 1e-12
