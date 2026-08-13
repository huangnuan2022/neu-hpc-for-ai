from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ENDPOINTS = tuple(f"http://127.0.0.1:{port}" for port in range(8101, 8105))


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class BackendConfig:
    backend_id: str
    url: str


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    backends: tuple[BackendConfig, ...] = tuple(
        BackendConfig(f"worker-{index}", url) for index, url in enumerate(DEFAULT_ENDPOINTS)
    )
    max_in_flight: int = 64
    request_timeout_seconds: float = 120.0
    health_interval_seconds: float = 5.0
    failover_attempts: int = 2
    unhealthy_after_failures: int = 2
    prefix_chars: int = 256
    affinity_load_slack: int = 2
    max_prefetch_bytes: int = 1_048_576

    @classmethod
    def from_env(cls) -> "Settings":
        raw_endpoints = os.getenv("SERVING_BACKEND_ENDPOINTS", ",".join(DEFAULT_ENDPOINTS))
        endpoints = tuple(item.strip().rstrip("/") for item in raw_endpoints.split(",") if item.strip())
        if not endpoints:
            raise ValueError("SERVING_BACKEND_ENDPOINTS must contain at least one URL")
        return cls(
            model=os.getenv("SERVING_MODEL", DEFAULT_MODEL),
            backends=tuple(BackendConfig(f"worker-{index}", url) for index, url in enumerate(endpoints)),
            max_in_flight=_positive_int("SERVING_MAX_IN_FLIGHT", 64),
            request_timeout_seconds=_positive_float("SERVING_REQUEST_TIMEOUT_SECONDS", 120.0),
            health_interval_seconds=_positive_float("SERVING_HEALTH_INTERVAL_SECONDS", 5.0),
            failover_attempts=_positive_int("SERVING_FAILOVER_ATTEMPTS", 2),
            unhealthy_after_failures=_positive_int("SERVING_UNHEALTHY_AFTER_FAILURES", 2),
            prefix_chars=_positive_int("SERVING_PREFIX_CHARS", 256),
            affinity_load_slack=int(os.getenv("SERVING_AFFINITY_LOAD_SLACK", "2")),
            max_prefetch_bytes=_positive_int("SERVING_MAX_PREFETCH_BYTES", 1_048_576),
        )
