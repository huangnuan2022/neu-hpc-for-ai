from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUEST_COLUMNS = (
    "deployment",
    "worker_count",
    "concurrency",
    "mode",
    "run_index",
    "request_index",
    "prefix_group",
    "status",
    "error",
    "http_status",
    "backend",
    "route_reason",
    "backend_outstanding_at_route",
    "queue_depth_at_route",
    "input_tokens",
    "requested_output_tokens",
    "output_tokens",
    "ttft_seconds",
    "tpot_seconds",
    "e2e_seconds",
)


def validate_artifact(artifact: dict[str, Any]) -> None:
    required = {"schema_version", "generated_at", "config", "environment", "summaries", "runs"}
    missing = required - artifact.keys()
    if missing:
        raise ValueError(f"benchmark artifact is missing fields: {sorted(missing)}")
    config = artifact["config"]
    if config.get("model") != "Qwen/Qwen3-8B":
        raise ValueError("benchmark artifact must use the single configured Qwen3-8B model")
    if config.get("backend_kind") == "fake" and artifact.get("performance_evidence_valid") is not False:
        raise ValueError("fake backend artifacts cannot be marked as valid performance evidence")
    if artifact.get("performance_evidence_valid"):
        environment = artifact["environment"]
        image_metadata = environment.get("vllm_image") or {}
        if config.get("backend_kind") != "vllm" or config.get("tokenizer_mode") != "huggingface":
            raise ValueError("valid performance evidence requires vLLM and the Qwen tokenizer")
        if environment.get("git_dirty") or not image_metadata.get("image_id"):
            raise ValueError("valid performance evidence requires a clean commit and inspected image")
        if len(environment.get("gpus", [])) < int(config.get("worker_count", 0)):
            raise ValueError("valid performance evidence requires metadata for every configured GPU worker")
    for summary in artifact["summaries"]:
        for field in (
            "concurrency",
            "request_count",
            "ttft_seconds",
            "tpot_seconds",
            "e2e_seconds",
            "aggregate_output_tokens_per_second",
            "error_rate",
        ):
            if field not in summary:
                raise ValueError(f"summary is missing {field}")


def _format(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(artifact: dict[str, Any]) -> str:
    config = artifact["config"]
    environment = artifact["environment"]
    evidence = (
        "Measured vLLM performance evidence."
        if artifact["performance_evidence_valid"]
        else "Local fake-backend validation only; timing and throughput are not performance evidence."
    )
    lines = [
        "# Serving Benchmark Report",
        "",
        f"- Evidence status: **{evidence}**",
        f"- Generated at: `{artifact['generated_at']}`",
        f"- Deployment: `{config['deployment']}`",
        f"- Model: `{config['model']}`",
        f"- Backend: `{config['backend_kind']}` with `{config['worker_count']}` worker(s)",
        f"- Workload: `{config['input_tokens']}` input tokens, `{config['output_tokens']}` requested output tokens",
        f"- Concurrency: `{config['concurrency']}`",
        f"- Runs: `{config['warmup_runs']}` warm-up + `{config['measured_runs']}` measured per concurrency",
        f"- Git SHA: `{environment.get('git_sha')}` (dirty: `{environment.get('git_dirty')}`)",
        f"- AWS: `{environment.get('aws')}`",
        f"- GPUs: `{environment.get('gpus')}`",
        f"- CUDA toolkit: `{environment.get('cuda_toolkit_version')}`",
        f"- CUDA driver API: `{environment.get('cuda_driver_api_version')}`",
        f"- vLLM image: `{environment.get('vllm_image')}`",
        "",
        "## Results",
        "",
        "| mode | concurrency | requests | output tok/s | req/s | p50 TTFT | p95 TTFT | p99 TTFT | p95 TPOT | p95 E2E | error | vs sequential | cost / 1M output tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in artifact["summaries"]:
        cost_value = summary.get("estimated_cost_per_million_output_tokens_usd")
        cost_text = "n/a" if cost_value is None else f"${_format(cost_value, 2)}"
        lines.append(
            "| {mode} | {concurrency} | {requests} | {tokens_s} | {requests_s} | {ttft50} s | {ttft95} s | {ttft99} s | {tpot95} s | {e2e95} s | {error} | {speedup}x | {cost} |".format(
                mode=summary["mode"],
                concurrency=summary["concurrency"],
                requests=summary["request_count"],
                tokens_s=_format(summary["aggregate_output_tokens_per_second"]),
                requests_s=_format(summary["requests_per_second"]),
                ttft50=_format(summary["ttft_seconds"]["p50"]),
                ttft95=_format(summary["ttft_seconds"]["p95"]),
                ttft99=_format(summary["ttft_seconds"]["p99"]),
                tpot95=_format(summary["tpot_seconds"]["p95"]),
                e2e95=_format(summary["e2e_seconds"]["p95"]),
                error=_format(summary["error_rate"] * 100, 2) + "%",
                speedup=_format(summary.get("throughput_vs_sequential"), 2),
                cost=cost_text,
            )
        )

    lines.extend(["", "## Routing And Cache-Affinity Proxy", ""])
    for summary in artifact["summaries"]:
        lines.append(
            f"- Concurrency `{summary['concurrency']}`: affinity-route rate `{_format(summary.get('affinity_route_rate', 0) * 100, 2)}%`, "
            f"backend distribution `{summary['backend_counts']}`, route reasons `{summary['route_reason_counts']}`."
        )

    lines.extend(["", "## GPU And vLLM Telemetry", ""])
    for summary in artifact["summaries"]:
        lines.append(
            f"- Concurrency `{summary['concurrency']}` GPU: `{summary.get('gpu_summary', {})}`; "
            f"vLLM: `{summary.get('vllm_metrics_summary', {}).get('fleet', {})}`."
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Concurrency 1 is the request-at-a-time sequential baseline; higher concurrency keeps the gateway continuously supplied.",
            "- TTFT starts before the HTTP request and ends at the first non-empty generated token event.",
            "- TPOT is `(E2E - TTFT) / (output_tokens - 1)` when at least two output tokens are observed.",
            "- Output token counts use the streamed usage record when available and the configured tokenizer as a fallback.",
            "- Cost per million output tokens uses measured wall time and the supplied instance hourly price; it excludes idle setup outside the measured windows.",
            "- Prefix-affinity rate is a routing proxy, not a direct vLLM KV-cache hit rate. Direct cache usage is recorded separately when worker metrics expose it.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(artifact: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    validate_artifact(artifact)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "serving_benchmark.json"
    csv_path = output_dir / "serving_requests.csv"
    markdown_path = output_dir / "serving_benchmark.md"

    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for run in artifact["runs"]:
            for record in run["requests"]:
                writer.writerow(record)
    markdown_path.write_text(_markdown(artifact), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}
