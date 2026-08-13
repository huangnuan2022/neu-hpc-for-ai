from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .benchmark_report import validate_artifact


def _scenario_key(summary: dict[str, Any]) -> tuple[int, int, str]:
    return int(summary["worker_count"]), int(summary["concurrency"]), str(summary["mode"])


def compare_artifacts(
    baseline: dict[str, Any],
    current: dict[str, Any],
    max_regression: float,
) -> list[str]:
    validate_artifact(baseline)
    validate_artifact(current)
    if not baseline["performance_evidence_valid"] or not current["performance_evidence_valid"]:
        raise ValueError("regression checks require two valid real-GPU vLLM artifacts")
    if baseline["config"]["model"] != current["config"]["model"]:
        raise ValueError("baseline and current model differ")
    if not 0 <= max_regression < 1:
        raise ValueError("max_regression must be in [0, 1)")

    baseline_summaries = {_scenario_key(summary): summary for summary in baseline["summaries"]}
    current_summaries = {_scenario_key(summary): summary for summary in current["summaries"]}
    if baseline_summaries.keys() != current_summaries.keys():
        raise ValueError("baseline and current scenario matrices differ")

    failures: list[str] = []
    for key in sorted(baseline_summaries):
        before = baseline_summaries[key]
        after = current_summaries[key]
        throughput_before = float(before["aggregate_output_tokens_per_second"])
        throughput_after = float(after["aggregate_output_tokens_per_second"])
        if throughput_after < throughput_before * (1.0 - max_regression):
            regression = 1.0 - throughput_after / throughput_before
            failures.append(
                f"{key}: output throughput regressed {regression:.2%} "
                f"({throughput_before:.3f} -> {throughput_after:.3f} tokens/s)"
            )

        for metric in ("ttft_seconds", "e2e_seconds"):
            before_p95 = float(before[metric]["p95"])
            after_p95 = float(after[metric]["p95"])
            if after_p95 > before_p95 * (1.0 + max_regression):
                regression = after_p95 / before_p95 - 1.0
                failures.append(
                    f"{key}: p95 {metric} regressed {regression:.2%} "
                    f"({before_p95:.6f}s -> {after_p95:.6f}s)"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two controlled serving benchmark artifacts")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--max-regression", type=float, default=0.10)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    failures = compare_artifacts(baseline, current, args.max_regression)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"PASS: no controlled metric regressed by more than {args.max_regression:.1%}")


if __name__ == "__main__":
    main()
