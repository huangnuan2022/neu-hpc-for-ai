from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from opentelemetry import trace

from llm_serving_system.backend import FakeBackendTransport, FakeBehavior, token_event
from llm_serving_system.config import BackendConfig, Settings
from llm_serving_system.gateway import GatewayRuntime, SSETokenDetector, create_app
from llm_serving_system.routing import BackendRegistry, stable_prefix


def settings(*, workers: int = 2, timeout: float = 1.0, max_in_flight: int = 8) -> Settings:
    return Settings(
        backends=tuple(BackendConfig(f"worker-{index}", f"http://worker-{index}") for index in range(workers)),
        request_timeout_seconds=timeout,
        health_interval_seconds=60.0,
        max_in_flight=max_in_flight,
        failover_attempts=workers,
        unhealthy_after_failures=1,
        affinity_load_slack=2,
    )


def request_body(prompt: str = "shared prefix") -> dict[str, object]:
    return {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 4,
    }


def preferred_worker(config: Settings, body: dict[str, object]) -> str:
    prefix = stable_prefix(body, config.prefix_chars)
    return max(
        config.backends,
        key=lambda backend: BackendRegistry._affinity_score(prefix, backend.backend_id),
    ).backend_id


def test_sse_token_detector_handles_split_events_and_role_chunks() -> None:
    detector = SSETokenDetector()
    role = b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
    token = b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
    assert detector.feed(role) is False
    assert detector.feed(token[:17]) is False
    assert detector.feed(token[17:]) is True


def test_streaming_request_exposes_route_and_metrics() -> None:
    config = settings()
    transport = FakeBackendTransport()
    app = create_app(config, transport, run_health_checks=False)

    with TestClient(app) as client:
        first = client.post("/v1/chat/completions", json=request_body())
        second = client.post("/v1/chat/completions", json=request_body())
        metrics = client.get("/metrics")

    assert first.status_code == 200
    assert first.headers["x-llm-backend"] == second.headers["x-llm-backend"]
    assert first.headers["x-llm-route-reason"] == "prefix_affinity"
    assert int(first.headers["x-llm-backend-outstanding"]) >= 1
    assert "fake-token" in first.text
    assert "llm_gateway_ttft_seconds_count" in metrics.text
    assert "llm_gateway_route_decisions_total" in metrics.text


def test_failure_before_first_token_fails_over_safely() -> None:
    config = settings()
    body = request_body("prefix that selects a failing worker")
    failing = preferred_worker(config, body)
    transport = FakeBackendTransport({failing: FakeBehavior(open_failures=1)})
    app = create_app(config, transport, run_health_checks=False)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["x-llm-backend"] != failing
    assert response.headers["x-llm-route-reason"].startswith("failover_")
    assert transport.open_calls[failing] == 1
    assert sum(transport.open_calls.values()) == 2


def test_midstream_failure_does_not_retry_another_worker() -> None:
    config = settings()
    body = request_body("prefix for a midstream failure")
    selected = preferred_worker(config, body)
    behavior = FakeBehavior(
        chunks=(token_event("first "), token_event("second "), b"data: [DONE]\n\n"),
        fail_after_chunks=1,
    )
    transport = FakeBackendTransport({selected: behavior})
    app = create_app(config, transport, run_health_checks=False)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)
        metrics = client.get("/metrics")

    assert response.status_code == 200
    assert "first" in response.text
    assert "second" not in response.text
    assert transport.open_calls == {selected: 1}
    assert 'stage="after_first_token"' in metrics.text


def test_deadline_before_first_token_returns_503() -> None:
    config = settings(workers=1, timeout=0.01)
    transport = FakeBackendTransport({"worker-0": FakeBehavior(initial_delay=0.1)})
    app = create_app(config, transport, run_health_checks=False)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=request_body())

    assert response.status_code == 503
    assert "worker-0" in response.json()["detail"]
    assert transport.close_calls["worker-0"] == 1


def test_empty_stream_fails_over_before_returning_to_client() -> None:
    config = settings()
    body = request_body("prefix for an empty stream")
    empty_worker = preferred_worker(config, body)
    transport = FakeBackendTransport(
        {empty_worker: FakeBehavior(chunks=(b"data: [DONE]\n\n",))}
    )
    app = create_app(config, transport, run_health_checks=False)

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.headers["x-llm-backend"] != empty_worker
    assert response.headers["x-llm-route-reason"].startswith("failover_")


def test_gateway_returns_429_when_admission_is_full() -> None:
    config = settings(workers=1, max_in_flight=1)
    app = create_app(config, FakeBackendTransport(), run_health_checks=False)

    with TestClient(app) as client:
        runtime = app.state.gateway
        assert client.portal.call(runtime.admission.try_acquire) is True
        response = client.post("/v1/chat/completions", json=request_body())
        client.portal.call(runtime.admission.release)

    assert response.status_code == 429
    assert response.json()["detail"] == "gateway admission limit reached"


def test_non_streaming_request_and_single_model_guard() -> None:
    config = settings(workers=1)
    transport = FakeBackendTransport()
    app = create_app(config, transport, run_health_checks=False)
    body = request_body()
    body["stream"] = False

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)
        wrong_model = dict(body, model="another/model")
        rejected = client.post("/v1/chat/completions", json=wrong_model)

    assert response.status_code == 200
    assert response.json()["model"] == "Qwen/Qwen3-8B"
    assert rejected.status_code == 400


def test_health_checks_can_mark_all_backends_unavailable() -> None:
    config = settings(workers=1)
    transport = FakeBackendTransport({"worker-0": FakeBehavior(healthy=False)})
    app = create_app(config, transport, run_health_checks=True)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"


def test_client_stream_cancellation_releases_backend_and_admission_slots() -> None:
    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def scenario() -> None:
        config = settings(workers=1)
        behavior = FakeBehavior(
            chunks=(token_event("first "), token_event("second "), b"data: [DONE]\n\n"),
            chunk_delay=0.01,
        )
        transport = FakeBackendTransport({"worker-0": behavior})
        runtime = GatewayRuntime(config, transport)
        assert await runtime.admission.try_acquire()
        runtime.metrics.in_flight.inc()
        started_at = asyncio.get_running_loop().time()
        request = ConnectedRequest()
        prepared = await runtime.prepare_stream(
            request,  # type: ignore[arg-type]
            "/v1/chat/completions",
            request_body(),
            "cancel-prefix",
            started_at,
        )
        span = trace.get_tracer(__name__).start_span("cancellation-test")
        stream = runtime.stream_body(prepared, request, "chat", span)  # type: ignore[arg-type]
        assert b"first" in await anext(stream)
        await stream.aclose()

        snapshot = await runtime.registry.snapshot()
        assert snapshot[0]["outstanding"] == 0
        assert await runtime.admission.current() == 0
        assert transport.close_calls["worker-0"] == 1

    asyncio.run(scenario())
