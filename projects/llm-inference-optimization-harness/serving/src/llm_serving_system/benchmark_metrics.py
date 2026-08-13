from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    collected = list(values)
    if not collected:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    return {
        "mean": sum(collected) / len(collected),
        "p50": percentile(collected, 50),
        "p95": percentile(collected, 95),
        "p99": percentile(collected, 99),
    }


def summarize_scenario(
    request_records: list[dict[str, Any]],
    measured_wall_seconds: float,
    hourly_cost_usd: float | None,
) -> dict[str, Any]:
    successful = [record for record in request_records if record["status"] == "ok"]
    timed_out = [record for record in request_records if record["status"] == "timeout"]
    failed = [record for record in request_records if record["status"] not in {"ok", "timeout"}]
    output_tokens = sum(int(record["output_tokens"]) for record in successful)
    backend_counts = Counter(str(record["backend"]) for record in successful if record.get("backend"))
    route_counts = Counter(str(record["route_reason"]) for record in successful if record.get("route_reason"))
    queue_depths = [float(record["queue_depth_at_route"]) for record in successful]

    cost_per_million: float | None = None
    estimated_cost: float | None = None
    if hourly_cost_usd is not None:
        estimated_cost = hourly_cost_usd * measured_wall_seconds / 3600.0
        if output_tokens > 0:
            cost_per_million = estimated_cost * 1_000_000.0 / output_tokens

    affinity_routes = sum(
        count for reason, count in route_counts.items() if "prefix_affinity" in reason
    )
    return {
        "request_count": len(request_records),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "timeout_requests": len(timed_out),
        "error_rate": (len(failed) + len(timed_out)) / len(request_records) if request_records else 0.0,
        "timeout_rate": len(timed_out) / len(request_records) if request_records else 0.0,
        "measured_wall_seconds": measured_wall_seconds,
        "output_tokens": output_tokens,
        "aggregate_output_tokens_per_second": output_tokens / measured_wall_seconds if measured_wall_seconds else None,
        "requests_per_second": len(successful) / measured_wall_seconds if measured_wall_seconds else None,
        "ttft_seconds": distribution(float(record["ttft_seconds"]) for record in successful),
        "tpot_seconds": distribution(
            float(record["tpot_seconds"])
            for record in successful
            if record.get("tpot_seconds") is not None
        ),
        "inter_chunk_seconds": distribution(
            float(interval)
            for record in successful
            for interval in record.get("inter_chunk_seconds", [])
        ),
        "e2e_seconds": distribution(float(record["e2e_seconds"]) for record in successful),
        "queue_depth_at_route": distribution(queue_depths),
        "backend_counts": dict(sorted(backend_counts.items())),
        "route_reason_counts": dict(sorted(route_counts.items())),
        "affinity_route_rate": affinity_routes / len(successful) if successful else None,
        "estimated_cost_usd": estimated_cost,
        "estimated_cost_per_million_output_tokens_usd": cost_per_million,
    }


def add_sequential_comparisons(summaries: list[dict[str, Any]]) -> None:
    baseline = next((summary for summary in summaries if summary["concurrency"] == 1), None)
    baseline_throughput = baseline.get("aggregate_output_tokens_per_second") if baseline else None
    for summary in summaries:
        throughput = summary.get("aggregate_output_tokens_per_second")
        summary["throughput_vs_sequential"] = (
            throughput / baseline_throughput
            if throughput is not None and baseline_throughput not in {None, 0}
            else None
        )
