from __future__ import annotations

import asyncio

from llm_serving_system.config import BackendConfig
from llm_serving_system.routing import AdmissionController, BackendRegistry, stable_prefix


BACKENDS = (
    BackendConfig("worker-0", "http://worker-0"),
    BackendConfig("worker-1", "http://worker-1"),
    BackendConfig("worker-2", "http://worker-2"),
    BackendConfig("worker-3", "http://worker-3"),
)


def test_stable_prefix_normalizes_whitespace_and_ignores_suffix() -> None:
    first = {"messages": [{"role": "user", "content": "shared   prefix then A"}]}
    second = {"messages": [{"role": "user", "content": "shared prefix then B"}]}
    assert stable_prefix(first, 13) == stable_prefix(second, 13)


def test_prefix_affinity_is_stable() -> None:
    async def scenario() -> None:
        registry = BackendRegistry(BACKENDS, affinity_load_slack=2, unhealthy_after_failures=2)
        first = await registry.reserve("stable system prompt")
        await registry.release(first.backend.backend_id, success=True)
        second = await registry.reserve("stable system prompt")
        assert second.backend.backend_id == first.backend.backend_id
        assert second.reason == "prefix_affinity"

    asyncio.run(scenario())

def test_affinity_yields_to_load_and_failover_excludes_backend() -> None:
    async def scenario() -> None:
        registry = BackendRegistry(BACKENDS, affinity_load_slack=0, unhealthy_after_failures=2)
        preferred = await registry.reserve("same prefix")
        second = await registry.reserve("same prefix")
        assert second.backend.backend_id != preferred.backend.backend_id
        assert second.reason == "affinity_overridden_by_load"
        await registry.release(second.backend.backend_id, success=True)
        failover = await registry.reserve("same prefix", {preferred.backend.backend_id})
        assert failover.backend.backend_id != preferred.backend.backend_id
        assert failover.reason.startswith("failover_")

    asyncio.run(scenario())


def test_admission_controller_rejects_without_waiting() -> None:
    async def scenario() -> None:
        admission = AdmissionController(1)
        assert await admission.try_acquire() is True
        assert await admission.try_acquire() is False
        assert await admission.current() == 1
        await admission.release()
        assert await admission.try_acquire() is True

    asyncio.run(scenario())
