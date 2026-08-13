from __future__ import annotations

import argparse
import asyncio
import codecs
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .benchmark_environment import GpuSampler, collect_environment, parse_vllm_metrics
from .benchmark_metrics import add_sequential_comparisons, summarize_scenario
from .benchmark_report import write_artifacts
from .config import DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parents[1]


class BenchmarkTokenizer(Protocol):
    name: str

    def prompt_tokens(self, token_count: int, prefix_group: int) -> list[int]:
        ...

    def count_text(self, text: str) -> int:
        ...


class WhitespaceTokenizer:
    name = "deterministic-whitespace-test-tokenizer"

    def prompt_tokens(self, token_count: int, prefix_group: int) -> list[int]:
        start = 10_000 + prefix_group * 1_000
        return [start + index % 997 for index in range(token_count)]

    def count_text(self, text: str) -> int:
        return len(text.split())


class HuggingFaceTokenizer:
    def __init__(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("install the benchmark extra: pip install -e '.[benchmark]'") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        self.name = str(getattr(self._tokenizer, "name_or_path", DEFAULT_MODEL))

    def prompt_tokens(self, token_count: int, prefix_group: int) -> list[int]:
        seed = (
            f"Benchmark prefix group {prefix_group}. "
            "Analyze distributed language model serving, request scheduling, and GPU execution. "
        )
        encoded = self._tokenizer.encode(seed, add_special_tokens=False)
        if not encoded:
            raise RuntimeError("tokenizer returned no prompt tokens")
        repeats = (token_count + len(encoded) - 1) // len(encoded)
        return (encoded * repeats)[:token_count]

    def count_text(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))


class SSEParser:
    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[dict[str, Any] | str]:
        self._buffer += self._decoder.decode(chunk).replace("\r\n", "\n")
        events: list[dict[str, Any] | str] = []
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
            if not data:
                continue
            if data == "[DONE]":
                events.append("[DONE]")
                continue
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events


def _event_content(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list):
        return ""
    parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            parts.append(delta["content"])
        elif isinstance(choice.get("text"), str):
            parts.append(choice["text"])
    return "".join(parts)


@dataclass(frozen=True)
class WorkloadConfig:
    endpoint: str
    deployment: str
    worker_count: int
    input_tokens: int
    output_tokens: int
    prefix_groups: int
    timeout_seconds: float


async def execute_request(
    client: httpx.AsyncClient,
    tokenizer: BenchmarkTokenizer,
    workload: WorkloadConfig,
    *,
    concurrency: int,
    run_index: int,
    request_index: int,
) -> dict[str, Any]:
    prefix_group = request_index % workload.prefix_groups
    prompt = tokenizer.prompt_tokens(workload.input_tokens, prefix_group)
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "max_tokens": workload.output_tokens,
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    mode = "sequential" if concurrency == 1 else "continuous"
    record: dict[str, Any] = {
        "deployment": workload.deployment,
        "worker_count": workload.worker_count,
        "concurrency": concurrency,
        "mode": mode,
        "run_index": run_index,
        "request_index": request_index,
        "prefix_group": prefix_group,
        "status": "error",
        "error": None,
        "http_status": None,
        "backend": None,
        "route_reason": None,
        "backend_outstanding_at_route": 0,
        "queue_depth_at_route": 0,
        "input_tokens": len(prompt),
        "requested_output_tokens": workload.output_tokens,
        "output_tokens": 0,
        "ttft_seconds": None,
        "tpot_seconds": None,
        "e2e_seconds": None,
        "inter_chunk_seconds": [],
    }

    started = time.perf_counter()
    content_parts: list[str] = []
    content_times: list[float] = []
    usage_output_tokens: int | None = None
    parser = SSEParser()
    try:
        async with asyncio.timeout(workload.timeout_seconds):
            async with client.stream(
                "POST",
                f"{workload.endpoint.rstrip('/')}/v1/completions",
                json=payload,
            ) as response:
                record["http_status"] = response.status_code
                record["backend"] = response.headers.get("x-llm-backend")
                record["route_reason"] = response.headers.get("x-llm-route-reason")
                outstanding = int(response.headers.get("x-llm-backend-outstanding", "0"))
                record["backend_outstanding_at_route"] = outstanding
                record["queue_depth_at_route"] = max(0, outstanding - 1)
                if response.status_code >= 400:
                    body = (await response.aread())[:512].decode("utf-8", errors="replace")
                    record["error"] = f"HTTP {response.status_code}: {body}"
                    record["e2e_seconds"] = time.perf_counter() - started
                    return record

                async for chunk in response.aiter_bytes():
                    for event in parser.feed(chunk):
                        if event == "[DONE]":
                            continue
                        usage = event.get("usage")
                        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                            usage_output_tokens = usage["completion_tokens"]
                        content = _event_content(event)
                        if content:
                            content_parts.append(content)
                            content_times.append(time.perf_counter())
    except TimeoutError:
        record["status"] = "timeout"
        record["error"] = f"request exceeded {workload.timeout_seconds:.3f}s deadline"
    except (httpx.HTTPError, OSError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    finished = time.perf_counter()
    record["e2e_seconds"] = finished - started
    if not content_times:
        if record["status"] != "timeout" and record["error"] is None:
            record["error"] = "stream ended without generated content"
        return record

    output_tokens = usage_output_tokens
    if output_tokens is None:
        output_tokens = tokenizer.count_text("".join(content_parts))
    record["output_tokens"] = output_tokens
    record["ttft_seconds"] = content_times[0] - started
    record["inter_chunk_seconds"] = [
        later - earlier for earlier, later in zip(content_times, content_times[1:], strict=False)
    ]
    if output_tokens > 1:
        record["tpot_seconds"] = max(0.0, (record["e2e_seconds"] - record["ttft_seconds"]) / (output_tokens - 1))
    if record["status"] != "timeout":
        record["status"] = "ok"
        record["error"] = None
    return record


async def run_batch(
    client: httpx.AsyncClient,
    tokenizer: BenchmarkTokenizer,
    workload: WorkloadConfig,
    *,
    concurrency: int,
    run_index: int,
    request_count: int,
) -> tuple[list[dict[str, Any]], float]:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(request_index: int) -> dict[str, Any]:
        async with semaphore:
            return await execute_request(
                client,
                tokenizer,
                workload,
                concurrency=concurrency,
                run_index=run_index,
                request_index=request_index,
            )

    started = time.perf_counter()
    records = await asyncio.gather(*(bounded(index) for index in range(request_count)))
    return records, time.perf_counter() - started


class VllmMetricsSampler:
    def __init__(self, urls: list[str], interval_seconds: float = 0.5) -> None:
        self.urls = urls
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        if not self.urls:
            return
        async with httpx.AsyncClient(timeout=2.0) as client:
            while not self._stop.is_set():
                responses = await asyncio.gather(
                    *(client.get(url) for url in self.urls),
                    return_exceptions=True,
                )
                timestamp = time.time()
                for url, response in zip(self.urls, responses, strict=True):
                    if isinstance(response, httpx.Response) and response.status_code < 400:
                        self.samples.append(
                            {"timestamp": timestamp, "url": url, "metrics": parse_vllm_metrics(response.text)}
                        )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()

    def summary(
        self,
        start_epoch: float | None = None,
        end_epoch: float | None = None,
    ) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[float]] = {}
        for sample in self.samples:
            if start_epoch is not None and sample["timestamp"] < start_epoch:
                continue
            if end_epoch is not None and sample["timestamp"] > end_epoch:
                continue
            for name, value in sample["metrics"].items():
                grouped.setdefault((sample["url"], name), []).append(float(value))

        workers: dict[str, dict[str, dict[str, float]]] = {}
        for (url, name), values in sorted(grouped.items()):
            workers.setdefault(url, {})[name] = {
                "sample_count": len(values),
                "first": values[0],
                "last": values[-1],
                "delta": values[-1] - values[0],
                "mean": sum(values) / len(values),
                "max": max(values),
            }

        hit_delta = 0.0
        query_delta = 0.0
        max_cache_usage: float | None = None
        waiting_by_timestamp: dict[float, float] = {}
        running_by_timestamp: dict[float, float] = {}
        for sample in self.samples:
            if start_epoch is not None and sample["timestamp"] < start_epoch:
                continue
            if end_epoch is not None and sample["timestamp"] > end_epoch:
                continue
            for name, value in sample["metrics"].items():
                if name.endswith("num_requests_waiting"):
                    waiting_by_timestamp[sample["timestamp"]] = (
                        waiting_by_timestamp.get(sample["timestamp"], 0.0) + float(value)
                    )
                elif name.endswith("num_requests_running"):
                    running_by_timestamp[sample["timestamp"]] = (
                        running_by_timestamp.get(sample["timestamp"], 0.0) + float(value)
                    )
        for metrics in workers.values():
            for name, values in metrics.items():
                if name.endswith(("prefix_cache_hits", "prefix_cache_hits_total")):
                    hit_delta += max(0.0, values["delta"])
                elif name.endswith(("prefix_cache_queries", "prefix_cache_queries_total")):
                    query_delta += max(0.0, values["delta"])
                elif name.endswith(("kv_cache_usage_perc", "gpu_cache_usage_perc")):
                    max_cache_usage = max(max_cache_usage or 0.0, values["max"])
        return {
            "workers": workers,
            "fleet": {
                "prefix_cache_hit_tokens": hit_delta,
                "prefix_cache_query_tokens": query_delta,
                "prefix_cache_hit_rate": hit_delta / query_delta if query_delta else None,
                "max_kv_cache_usage": max_cache_usage,
                "max_requests_waiting_sum": max(waiting_by_timestamp.values(), default=0.0),
                "max_requests_running_sum": max(running_by_timestamp.values(), default=0.0),
            },
        }


def _parse_concurrency(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("concurrency must contain positive comma-separated integers")
    if 1 not in values:
        raise argparse.ArgumentTypeError("concurrency matrix must include the sequential baseline 1")
    return values


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer: BenchmarkTokenizer = (
        HuggingFaceTokenizer() if args.tokenizer == "huggingface" else WhitespaceTokenizer()
    )
    workload = WorkloadConfig(
        endpoint=args.endpoint,
        deployment=args.deployment,
        worker_count=args.worker_count,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        prefix_groups=args.prefix_groups,
        timeout_seconds=args.timeout_seconds,
    )
    gpu_sampler = GpuSampler(args.sample_interval_seconds)
    vllm_sampler = VllmMetricsSampler(args.backend_metrics_url, args.sample_interval_seconds)
    gpu_task = asyncio.create_task(gpu_sampler.run())
    vllm_task = asyncio.create_task(vllm_sampler.run())
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    limits = httpx.Limits(max_connections=max(args.concurrency) * 2, max_keepalive_connections=max(args.concurrency))
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        for concurrency in args.concurrency:
            request_count = max(concurrency, concurrency * args.requests_per_concurrency)
            for warmup_index in range(args.warmup_runs):
                await run_batch(
                    client,
                    tokenizer,
                    workload,
                    concurrency=concurrency,
                    run_index=-(warmup_index + 1),
                    request_count=request_count,
                )

            scenario_records: list[dict[str, Any]] = []
            scenario_wall = 0.0
            measurement_start_epoch = time.time()
            for run_index in range(args.measured_runs):
                records, wall_seconds = await run_batch(
                    client,
                    tokenizer,
                    workload,
                    concurrency=concurrency,
                    run_index=run_index,
                    request_count=request_count,
                )
                runs.append(
                    {
                        "concurrency": concurrency,
                        "mode": "sequential" if concurrency == 1 else "continuous",
                        "run_index": run_index,
                        "wall_seconds": wall_seconds,
                        "requests": records,
                    }
                )
                scenario_records.extend(records)
                scenario_wall += wall_seconds
            measurement_end_epoch = time.time()

            summary = summarize_scenario(scenario_records, scenario_wall, args.hourly_cost_usd)
            summary.update(
                {
                    "deployment": args.deployment,
                    "worker_count": args.worker_count,
                    "concurrency": concurrency,
                    "mode": "sequential" if concurrency == 1 else "continuous",
                    "measured_runs": args.measured_runs,
                    "requests_per_run": request_count,
                    "measurement_start_epoch": measurement_start_epoch,
                    "measurement_end_epoch": measurement_end_epoch,
                }
            )
            summaries.append(summary)

    gpu_sampler.stop()
    vllm_sampler.stop()
    await asyncio.gather(gpu_task, vllm_task)
    add_sequential_comparisons(summaries)
    for summary in summaries:
        start_epoch = summary.pop("measurement_start_epoch")
        end_epoch = summary.pop("measurement_end_epoch")
        summary["gpu_summary"] = gpu_sampler.summary(start_epoch, end_epoch)
        summary["vllm_metrics_summary"] = vllm_sampler.summary(start_epoch, end_epoch)

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    environment = await asyncio.to_thread(collect_environment, REPO_ROOT, args.vllm_image)
    image_metadata = environment.get("vllm_image") or {}
    all_requests = [record for run in runs for record in run["requests"]]
    performance_evidence_valid = (
        args.backend_kind == "vllm"
        and args.tokenizer == "huggingface"
        and len(environment["gpus"]) >= args.worker_count
        and not environment["git_dirty"]
        and bool(image_metadata.get("image_id"))
        and all_requests
        and all(
            record["status"] == "ok"
            and record["input_tokens"] == args.input_tokens
            and record["output_tokens"] == args.output_tokens
            for record in all_requests
        )
        and all(
            summary["successful_requests"] == summary["request_count"]
            and summary["output_tokens"] > 0
            for summary in summaries
        )
    )
    return {
        "schema_version": "1.0",
        "generated_at": generated_at,
        "performance_evidence_valid": performance_evidence_valid,
        "config": {
            "model": DEFAULT_MODEL,
            "endpoint": args.endpoint,
            "deployment": args.deployment,
            "worker_count": args.worker_count,
            "backend_kind": args.backend_kind,
            "tokenizer_mode": args.tokenizer,
            "tokenizer_name": tokenizer.name,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": args.concurrency,
            "prefix_groups": args.prefix_groups,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.measured_runs,
            "requests_per_concurrency": args.requests_per_concurrency,
            "timeout_seconds": args.timeout_seconds,
            "hourly_cost_usd": args.hourly_cost_usd,
            "backend_metrics_urls": args.backend_metrics_url,
            "vllm_image": args.vllm_image,
        },
        "environment": environment,
        "summaries": summaries,
        "runs": runs,
        "gpu_summary": gpu_sampler.summary(),
        "gpu_samples": gpu_sampler.samples,
        "vllm_metrics_summary": vllm_sampler.summary(),
        "vllm_metrics_samples": vllm_sampler.samples,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Run the reproducible Qwen3-8B serving load harness")
    command.add_argument("--endpoint", default="http://127.0.0.1:8000")
    command.add_argument("--deployment", default="local-fake")
    command.add_argument("--worker-count", type=int, choices=(1, 4), default=4)
    command.add_argument("--backend-kind", choices=("fake", "vllm"), default="fake")
    command.add_argument("--tokenizer", choices=("whitespace", "huggingface"), default="whitespace")
    command.add_argument("--input-tokens", type=int, default=512)
    command.add_argument("--output-tokens", type=int, default=128)
    command.add_argument("--concurrency", type=_parse_concurrency, default=_parse_concurrency("1,8,16,32"))
    command.add_argument("--prefix-groups", type=int, default=4)
    command.add_argument("--warmup-runs", type=int, default=1)
    command.add_argument("--measured-runs", type=int, default=5)
    command.add_argument("--requests-per-concurrency", type=int, default=1)
    command.add_argument("--timeout-seconds", type=float, default=180.0)
    command.add_argument("--hourly-cost-usd", type=float, default=None)
    command.add_argument("--backend-metrics-url", action="append", default=[])
    command.add_argument("--vllm-image", default=os.getenv("VLLM_IMAGE"))
    command.add_argument("--sample-interval-seconds", type=float, default=0.5)
    command.add_argument("--output-dir", type=Path, default=None)
    return command


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "input_tokens",
        "output_tokens",
        "prefix_groups",
        "warmup_runs",
        "measured_runs",
        "requests_per_concurrency",
    ):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.timeout_seconds <= 0 or args.sample_interval_seconds <= 0:
        raise SystemExit("timeouts and sample intervals must be positive")
    if args.backend_kind == "vllm" and args.tokenizer != "huggingface":
        raise SystemExit("real vLLM evidence requires --tokenizer huggingface")
    if args.backend_kind == "vllm" and args.hourly_cost_usd is None:
        raise SystemExit("real vLLM evidence requires --hourly-cost-usd for cost reporting")
    if args.backend_kind == "vllm" and (
        not args.vllm_image or args.vllm_image.endswith(":latest")
    ):
        raise SystemExit("real vLLM evidence requires a pinned --vllm-image tag or digest")
    if args.measured_runs < 5:
        raise SystemExit("at least five measured runs are required")


def main() -> None:
    args = parser().parse_args()
    _validate_args(args)
    if not args.backend_metrics_url and args.backend_kind == "vllm":
        endpoints = [value.strip().rstrip("/") for value in os.getenv("SERVING_BACKEND_ENDPOINTS", "").split(",") if value.strip()]
        args.backend_metrics_url = [f"{endpoint}/metrics" for endpoint in endpoints]
    artifact = asyncio.run(run_benchmark(args))
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or PROJECT_ROOT / "serving-results" / f"{args.deployment}-{timestamp}"
    paths = write_artifacts(artifact, output_dir)
    print(f"Wrote benchmark artifacts to {output_dir}")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")


if __name__ == "__main__":
    main()
