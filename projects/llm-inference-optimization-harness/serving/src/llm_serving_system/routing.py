from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass

from .config import BackendConfig


_WHITESPACE = re.compile(r"\s+")


def stable_prefix(payload: dict[str, object], prefix_chars: int) -> str:
    if "messages" in payload:
        parts: list[str] = []
        messages = payload.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    parts.append(str(message.get("role", "")))
                    parts.append(str(message.get("content", "")))
        text = "\n".join(parts)
    else:
        prompt = payload.get("prompt", "")
        text = json.dumps(prompt, sort_keys=True) if not isinstance(prompt, str) else prompt
    return _WHITESPACE.sub(" ", text).strip()[:prefix_chars]


@dataclass
class BackendState:
    config: BackendConfig
    healthy: bool = True
    outstanding: int = 0
    completed: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None


@dataclass(frozen=True)
class RouteSelection:
    backend: BackendConfig
    reason: str
    outstanding: int


class NoHealthyBackend(RuntimeError):
    pass


class BackendRegistry:
    def __init__(
        self,
        backends: tuple[BackendConfig, ...],
        *,
        affinity_load_slack: int,
        unhealthy_after_failures: int,
    ) -> None:
        self._states = {backend.backend_id: BackendState(backend) for backend in backends}
        self._affinity_load_slack = max(0, affinity_load_slack)
        self._unhealthy_after_failures = unhealthy_after_failures
        self._lock = asyncio.Lock()

    @staticmethod
    def _affinity_score(prefix: str, backend_id: str) -> int:
        digest = hashlib.sha256(f"{prefix}\0{backend_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    async def reserve(self, prefix: str, excluded: set[str] | None = None) -> RouteSelection:
        excluded = excluded or set()
        async with self._lock:
            candidates = [
                state
                for state in self._states.values()
                if state.healthy and state.config.backend_id not in excluded
            ]
            if not candidates:
                raise NoHealthyBackend("no healthy backend is available")

            min_load = min(state.outstanding for state in candidates)
            least_loaded = min(candidates, key=lambda state: (state.outstanding, state.config.backend_id))
            selected = least_loaded
            reason = "least_loaded"

            if prefix:
                preferred = max(
                    candidates,
                    key=lambda state: self._affinity_score(prefix, state.config.backend_id),
                )
                if preferred.outstanding <= min_load + self._affinity_load_slack:
                    selected = preferred
                    reason = "prefix_affinity"
                else:
                    reason = "affinity_overridden_by_load"

            if excluded:
                reason = f"failover_{reason}"
            selected.outstanding += 1
            return RouteSelection(selected.config, reason, selected.outstanding)

    async def release(self, backend_id: str, *, success: bool, error: str | None = None) -> int:
        async with self._lock:
            state = self._states[backend_id]
            state.outstanding = max(0, state.outstanding - 1)
            if success:
                state.completed += 1
                state.consecutive_failures = 0
                state.last_error = None
            else:
                state.consecutive_failures += 1
                state.last_error = error
                if state.consecutive_failures >= self._unhealthy_after_failures:
                    state.healthy = False
            return state.outstanding

    async def set_health(self, backend_id: str, healthy: bool, error: str | None = None) -> None:
        async with self._lock:
            state = self._states[backend_id]
            state.healthy = healthy
            state.last_error = error
            if healthy:
                state.consecutive_failures = 0

    async def snapshot(self) -> list[dict[str, object]]:
        async with self._lock:
            return [
                {
                    "backend_id": state.config.backend_id,
                    "url": state.config.url,
                    "healthy": state.healthy,
                    "outstanding": state.outstanding,
                    "completed": state.completed,
                    "consecutive_failures": state.consecutive_failures,
                    "last_error": state.last_error,
                }
                for state in self._states.values()
            ]


class AdmissionController:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._in_flight = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._in_flight >= self.capacity:
                return False
            self._in_flight += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    async def current(self) -> int:
        async with self._lock:
            return self._in_flight
