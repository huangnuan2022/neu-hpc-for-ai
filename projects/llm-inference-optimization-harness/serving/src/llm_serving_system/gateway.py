from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from opentelemetry.trace import Span
from prometheus_client import CONTENT_TYPE_LATEST

from .backend import BackendError, BackendTransport, HttpBackendTransport, StreamHandle
from .config import Settings
from .metrics import ServingMetrics
from .routing import AdmissionController, BackendRegistry, NoHealthyBackend, RouteSelection, stable_prefix
from .telemetry import configure_tracing, tracer


class SSETokenDetector:
    """Detect the first generated token without assuming HTTP chunk boundaries."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> bool:
        self._buffer += self._decoder.decode(chunk).replace("\r\n", "\n")
        while "\n\n" in self._buffer:
            event, self._buffer = self._buffer.split("\n\n", 1)
            data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
            if not data or data == "[DONE]":
                continue
            try:
                body = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = body.get("choices") if isinstance(body, dict) else None
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if isinstance(delta, dict) and delta.get("content"):
                    return True
                if choice.get("text"):
                    return True
        return False


@dataclass
class PreparedStream:
    selection: RouteSelection
    handle: StreamHandle
    initial_chunks: list[bytes]
    started_at: float
    deadline: float


class GatewayRuntime:
    def __init__(self, settings: Settings, transport: BackendTransport | None = None) -> None:
        self.settings = settings
        self.transport = transport or HttpBackendTransport()
        self.registry = BackendRegistry(
            settings.backends,
            affinity_load_slack=settings.affinity_load_slack,
            unhealthy_after_failures=settings.unhealthy_after_failures,
        )
        self.admission = AdmissionController(settings.max_in_flight)
        self.metrics = ServingMetrics()
        self._stop = asyncio.Event()
        self._health_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._health_task = asyncio.create_task(self._health_loop(), name="backend-health-checks")

    async def stop(self) -> None:
        self._stop.set()
        if self._health_task:
            await self._health_task
        await self.transport.aclose()

    async def _health_loop(self) -> None:
        while not self._stop.is_set():
            results = await asyncio.gather(
                *(self.transport.health(backend) for backend in self.settings.backends),
                return_exceptions=True,
            )
            for backend, result in zip(self.settings.backends, results, strict=True):
                healthy = result is True
                error = None if healthy else "health check failed"
                if isinstance(result, Exception):
                    error = str(result)
                await self.registry.set_health(backend.backend_id, healthy, error)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.health_interval_seconds)
            except TimeoutError:
                pass

    def _record_selection(self, selection: RouteSelection) -> None:
        self.metrics.route_decisions.labels(selection.backend.backend_id, selection.reason).inc()
        self.metrics.backend_outstanding.labels(selection.backend.backend_id).set(selection.outstanding)

    async def _release_backend(
        self,
        selection: RouteSelection,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        outstanding = await self.registry.release(selection.backend.backend_id, success=success, error=error)
        self.metrics.backend_outstanding.labels(selection.backend.backend_id).set(outstanding)

    async def prepare_stream(
        self,
        request: Request,
        path: str,
        payload: dict[str, object],
        prefix: str,
        started_at: float,
    ) -> PreparedStream:
        deadline = started_at + self.settings.request_timeout_seconds
        excluded: set[str] = set()
        errors: list[str] = []
        attempts = min(self.settings.failover_attempts, len(self.settings.backends))

        for _ in range(attempts):
            selection = await self.registry.reserve(prefix, excluded)
            self._record_selection(selection)
            handle: StreamHandle | None = None
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request deadline expired before backend selection")
                async with asyncio.timeout(remaining):
                    handle = await self.transport.open_stream(selection.backend, path, payload)
                    detector = SSETokenDetector()
                    chunks: list[bytes] = []
                    buffered_bytes = 0
                    first_token_seen = False
                    async for chunk in handle.iterator:
                        if await request.is_disconnected():
                            raise asyncio.CancelledError
                        chunks.append(chunk)
                        buffered_bytes += len(chunk)
                        if buffered_bytes > self.settings.max_prefetch_bytes:
                            raise BackendError("backend exceeded the pre-token buffer limit")
                        if detector.feed(chunk):
                            first_token_seen = True
                            self.metrics.ttft.labels(selection.backend.backend_id).observe(
                                time.monotonic() - started_at
                            )
                            break
                    if not first_token_seen:
                        raise BackendError("backend stream ended before the first generated token")
                    return PreparedStream(selection, handle, chunks, started_at, deadline)
            except asyncio.CancelledError:
                if handle:
                    await handle.aclose()
                await self._release_backend(selection, success=True)
                raise
            except Exception as exc:
                if handle:
                    await handle.aclose()
                message = str(exc) or type(exc).__name__
                errors.append(f"{selection.backend.backend_id}: {message}")
                self.metrics.backend_errors.labels(selection.backend.backend_id, "before_first_token").inc()
                await self._release_backend(selection, success=False, error=message)
                excluded.add(selection.backend.backend_id)

        detail = "; ".join(errors) if errors else "no backend accepted the request"
        raise NoHealthyBackend(detail)

    async def complete(
        self,
        path: str,
        payload: dict[str, object],
        prefix: str,
        started_at: float,
    ) -> tuple[dict[str, object], RouteSelection]:
        deadline = started_at + self.settings.request_timeout_seconds
        excluded: set[str] = set()
        errors: list[str] = []
        attempts = min(self.settings.failover_attempts, len(self.settings.backends))

        for _ in range(attempts):
            selection = await self.registry.reserve(prefix, excluded)
            self._record_selection(selection)
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("request deadline expired before backend selection")
                async with asyncio.timeout(remaining):
                    data = await self.transport.complete(selection.backend, path, payload)
                await self._release_backend(selection, success=True)
                return data, selection
            except Exception as exc:
                message = str(exc) or type(exc).__name__
                errors.append(f"{selection.backend.backend_id}: {message}")
                self.metrics.backend_errors.labels(selection.backend.backend_id, "non_streaming").inc()
                await self._release_backend(selection, success=False, error=message)
                excluded.add(selection.backend.backend_id)

        raise NoHealthyBackend("; ".join(errors) if errors else "no backend accepted the request")

    async def stream_body(
        self,
        prepared: PreparedStream,
        request: Request,
        endpoint: str,
        span: Span,
    ) -> AsyncIterator[bytes]:
        selection = prepared.selection
        result = "ok"
        backend_success = True
        error: str | None = None
        try:
            for chunk in prepared.initial_chunks:
                yield chunk

            remaining = prepared.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request deadline expired during streaming")
            async with asyncio.timeout(remaining):
                async for chunk in prepared.handle.iterator:
                    if await request.is_disconnected():
                        result = "cancelled"
                        break
                    yield chunk
        except TimeoutError as exc:
            result = "timeout"
            error = str(exc) or "request deadline expired"
            self.metrics.backend_errors.labels(selection.backend.backend_id, "stream_timeout").inc()
        except asyncio.CancelledError:
            result = "cancelled"
            raise
        except Exception as exc:
            result = "stream_error"
            backend_success = False
            error = str(exc) or type(exc).__name__
            self.metrics.backend_errors.labels(selection.backend.backend_id, "after_first_token").inc()
            span.record_exception(exc)
        finally:
            await prepared.handle.aclose()
            await self._release_backend(selection, success=backend_success, error=error)
            self.metrics.requests.labels(endpoint, result).inc()
            self.metrics.e2e.labels(selection.backend.backend_id, result).observe(
                time.monotonic() - prepared.started_at
            )
            self.metrics.in_flight.dec()
            await self.admission.release()
            span.set_attribute("llm.result", result)
            span.end()


def _validated_payload(raw: object, path: str, settings: Settings) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="request body must be a JSON object")
    payload = dict(raw)
    requested_model = payload.get("model", settings.model)
    if requested_model != settings.model:
        raise HTTPException(
            status_code=400,
            detail=f"this deployment serves only {settings.model}",
        )
    payload["model"] = settings.model

    if path.endswith("chat/completions") and not isinstance(payload.get("messages"), list):
        raise HTTPException(status_code=422, detail="messages must be a list")
    if path.endswith("/completions") and not path.endswith("chat/completions") and "prompt" not in payload:
        raise HTTPException(status_code=422, detail="prompt is required")
    max_tokens = payload.get("max_tokens", 128)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise HTTPException(status_code=422, detail="max_tokens must be a positive integer")
    payload["max_tokens"] = max_tokens
    payload["stream"] = bool(payload.get("stream", True))
    return payload


def create_app(
    settings: Settings | None = None,
    transport: BackendTransport | None = None,
    *,
    run_health_checks: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()
    runtime = GatewayRuntime(settings, transport)
    configure_tracing()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if run_health_checks:
            await runtime.start()
        try:
            yield
        finally:
            if run_health_checks:
                await runtime.stop()
            else:
                await runtime.transport.aclose()

    app = FastAPI(title="Distributed LLM Serving & GPU Optimization System", lifespan=lifespan)
    app.state.gateway = runtime

    @app.get("/health")
    async def health() -> JSONResponse:
        backends = await runtime.registry.snapshot()
        ready = any(bool(backend["healthy"]) for backend in backends)
        return JSONResponse(
            {"status": "ready" if ready else "unavailable", "model": settings.model, "backends": backends},
            status_code=200 if ready else 503,
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(runtime.metrics.render(), media_type=CONTENT_TYPE_LATEST)

    async def proxy(request: Request, path: str) -> Response:
        try:
            raw = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="request body is not valid JSON") from exc
        payload = _validated_payload(raw, path, settings)
        prefix = stable_prefix(payload, settings.prefix_chars)
        endpoint = "chat" if path.endswith("chat/completions") else "completion"
        started_at = time.monotonic()

        if not await runtime.admission.try_acquire():
            runtime.metrics.requests.labels(endpoint, "rejected").inc()
            raise HTTPException(status_code=429, detail="gateway admission limit reached")
        runtime.metrics.in_flight.inc()

        span: Span = tracer().start_span("llm.gateway.request")
        span.set_attribute("llm.model", settings.model)
        span.set_attribute("llm.endpoint", endpoint)

        if not payload["stream"]:
            try:
                data, selection = await runtime.complete(path, payload, prefix, started_at)
                elapsed = time.monotonic() - started_at
                runtime.metrics.requests.labels(endpoint, "ok").inc()
                runtime.metrics.e2e.labels(selection.backend.backend_id, "ok").observe(elapsed)
                span.set_attribute("llm.backend", selection.backend.backend_id)
                span.set_attribute("llm.route_reason", selection.reason)
                return JSONResponse(
                    data,
                    headers={
                        "x-llm-backend": selection.backend.backend_id,
                        "x-llm-route-reason": selection.reason,
                    },
                )
            except NoHealthyBackend as exc:
                runtime.metrics.requests.labels(endpoint, "unavailable").inc()
                span.record_exception(exc)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            finally:
                span.end()
                runtime.metrics.in_flight.dec()
                await runtime.admission.release()

        try:
            prepared = await runtime.prepare_stream(request, path, payload, prefix, started_at)
        except NoHealthyBackend as exc:
            runtime.metrics.requests.labels(endpoint, "unavailable").inc()
            span.record_exception(exc)
            span.end()
            runtime.metrics.in_flight.dec()
            await runtime.admission.release()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except asyncio.CancelledError:
            runtime.metrics.requests.labels(endpoint, "cancelled").inc()
            span.end()
            runtime.metrics.in_flight.dec()
            await runtime.admission.release()
            raise

        selection = prepared.selection
        span.set_attribute("llm.backend", selection.backend.backend_id)
        span.set_attribute("llm.route_reason", selection.reason)

        return StreamingResponse(
            runtime.stream_body(prepared, request, endpoint, span),
            media_type="text/event-stream",
            headers={
                "x-llm-backend": selection.backend.backend_id,
                "x-llm-route-reason": selection.reason,
                "cache-control": "no-cache",
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await proxy(request, "/v1/completions")

    return app


def create_app_from_env() -> FastAPI:
    return create_app(Settings.from_env())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3-8B serving gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app_from_env(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
