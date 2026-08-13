from __future__ import annotations

import asyncio
import json
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _imds(path: str, token: str) -> str | None:
    request = urllib.request.Request(
        f"http://169.254.169.254/latest/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.2) as response:
            return response.read().decode("utf-8").strip()
    except (OSError, urllib.error.URLError):
        return None


def aws_instance_metadata() -> dict[str, str | None]:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    try:
        with urllib.request.urlopen(token_request, timeout=0.2) as response:
            token = response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return {"instance_id": None, "instance_type": None, "ami_id": None, "region": None}

    identity_text = _imds("dynamic/instance-identity/document", token)
    identity = json.loads(identity_text) if identity_text else {}
    return {
        "instance_id": _imds("meta-data/instance-id", token),
        "instance_type": _imds("meta-data/instance-type", token),
        "ami_id": _imds("meta-data/ami-id", token),
        "region": identity.get("region"),
    }


def collect_environment(repo_root: Path, vllm_image: str | None = None) -> dict[str, Any]:
    nvidia_query = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    if nvidia_query:
        for line in nvidia_query.splitlines():
            index, name, driver, memory_total = (part.strip() for part in line.split(",", 3))
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "driver_version": driver,
                    "memory_total_mib": float(memory_total),
                }
            )
    nvcc = _command_output(["nvcc", "--version"])
    cuda_match = re.search(r"release\s+([0-9.]+)", nvcc or "")
    nvidia_smi_full = _command_output(["nvidia-smi"])
    driver_cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_full or "")
    image_metadata: dict[str, Any] | None = None
    if vllm_image:
        image_id = _command_output(["docker", "image", "inspect", vllm_image, "--format", "{{.Id}}"])
        repo_digests_text = _command_output(
            ["docker", "image", "inspect", vllm_image, "--format", "{{json .RepoDigests}}"]
        )
        try:
            repo_digests = json.loads(repo_digests_text) if repo_digests_text else []
        except json.JSONDecodeError:
            repo_digests = []
        image_metadata = {"requested": vllm_image, "image_id": image_id, "repo_digests": repo_digests}
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_sha": _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_dirty": bool(_command_output(["git", "-C", str(repo_root), "status", "--porcelain"])),
        "cuda_toolkit_version": cuda_match.group(1) if cuda_match else None,
        "cuda_driver_api_version": driver_cuda_match.group(1) if driver_cuda_match else None,
        "nvcc": nvcc,
        "gpus": gpus,
        "aws": aws_instance_metadata(),
        "vllm_image": image_metadata,
    }


@dataclass
class GpuSampler:
    interval_seconds: float = 0.5
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def available(self) -> bool:
        return shutil.which("nvidia-smi") is not None

    async def run(self) -> None:
        if not self.available:
            return
        while not self._stop.is_set():
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                timestamp = time.time()
                for line in stdout.decode("utf-8").splitlines():
                    index, utilization, memory_used, memory_total = (
                        part.strip() for part in line.split(",", 3)
                    )
                    self.samples.append(
                        {
                            "timestamp": timestamp,
                            "gpu_index": int(index),
                            "utilization_pct": float(utilization),
                            "memory_used_mib": float(memory_used),
                            "memory_total_mib": float(memory_total),
                        }
                    )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    def summary(self, start_epoch: float | None = None, end_epoch: float | None = None) -> dict[str, Any]:
        by_gpu: dict[int, list[dict[str, Any]]] = {}
        for sample in self.samples:
            if start_epoch is not None and sample["timestamp"] < start_epoch:
                continue
            if end_epoch is not None and sample["timestamp"] > end_epoch:
                continue
            by_gpu.setdefault(int(sample["gpu_index"]), []).append(sample)
        return {
            str(index): {
                "sample_count": len(samples),
                "mean_utilization_pct": sum(sample["utilization_pct"] for sample in samples) / len(samples),
                "max_utilization_pct": max(sample["utilization_pct"] for sample in samples),
                "max_memory_used_mib": max(sample["memory_used_mib"] for sample in samples),
            }
            for index, samples in sorted(by_gpu.items())
        }


PROMETHEUS_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
VLLM_METRIC_SUFFIXES = (
    "gpu_cache_usage_perc",
    "kv_cache_usage_perc",
    "num_requests_running",
    "num_requests_waiting",
    "gpu_prefix_cache_hit_rate",
    "prefix_cache_hits",
    "prefix_cache_queries",
    "prompt_tokens_cached",
    "prompt_tokens",
    "prompt_tokens_total",
    "generation_tokens",
    "generation_tokens_total",
    "prefix_cache_hits_total",
    "prefix_cache_queries_total",
)


def parse_vllm_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        match = PROMETHEUS_SAMPLE.match(line.strip())
        if not match:
            continue
        name = match.group("name")
        if name.endswith(VLLM_METRIC_SUFFIXES):
            values[name] = values.get(name, 0.0) + float(match.group("value"))
    return values
