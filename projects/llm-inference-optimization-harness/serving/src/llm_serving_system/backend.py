from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import BackendConfig


class BackendError(RuntimeError):
    pass


@dataclass
class StreamHandle:
    iterator: AsyncIterator[bytes]
    close_callback: Callable[[], Awaitable[None]]

    async def aclose(self) -> None:
        await self.close_callback()


class BackendTransport(Protocol):
    async def open_stream(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> StreamHandle:
        ...

    async def complete(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> dict[str, object]:
        ...

    async def health(self, backend: BackendConfig) -> bool:
        ...

    async def aclose(self) -> None:
        ...


class HttpBackendTransport:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))

    async def open_stream(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> StreamHandle:
        request = self._client.build_request("POST", f"{backend.url}{path}", json=payload)
        response = await self._client.send(request, stream=True)
        if response.status_code >= 400:
            body = (await response.aread())[:512].decode("utf-8", errors="replace")
            await response.aclose()
            raise BackendError(f"{backend.backend_id} returned {response.status_code}: {body}")
        return StreamHandle(response.aiter_bytes(), response.aclose)

    async def complete(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> dict[str, object]:
        response = await self._client.post(f"{backend.url}{path}", json=payload)
        if response.status_code >= 400:
            raise BackendError(f"{backend.backend_id} returned {response.status_code}: {response.text[:512]}")
        data = response.json()
        if not isinstance(data, dict):
            raise BackendError(f"{backend.backend_id} returned a non-object JSON response")
        return data

    async def health(self, backend: BackendConfig) -> bool:
        try:
            response = await self._client.get(f"{backend.url}/health", timeout=3.0)
            return response.status_code < 400
        except httpx.HTTPError:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()


def token_event(content: str, request_id: str = "fake-request") -> bytes:
    body = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(body, separators=(',', ':'))}\n\n".encode("utf-8")


@dataclass
class FakeBehavior:
    chunks: tuple[bytes, ...] = (token_event("fake-token "), b"data: [DONE]\n\n")
    initial_delay: float = 0.0
    chunk_delay: float = 0.0
    open_failures: int = 0
    fail_after_chunks: int | None = None
    healthy: bool = True


@dataclass
class FakeBackendTransport:
    behaviors: dict[str, FakeBehavior] = field(default_factory=dict)
    open_calls: dict[str, int] = field(default_factory=dict)
    close_calls: dict[str, int] = field(default_factory=dict)

    def _behavior(self, backend: BackendConfig) -> FakeBehavior:
        return self.behaviors.setdefault(backend.backend_id, FakeBehavior())

    async def open_stream(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> StreamHandle:
        del path, payload
        behavior = self._behavior(backend)
        self.open_calls[backend.backend_id] = self.open_calls.get(backend.backend_id, 0) + 1
        if behavior.open_failures > 0:
            behavior.open_failures -= 1
            raise BackendError(f"injected open failure for {backend.backend_id}")

        closed = False

        async def chunks() -> AsyncIterator[bytes]:
            if behavior.initial_delay:
                await asyncio.sleep(behavior.initial_delay)
            for index, chunk in enumerate(behavior.chunks):
                if behavior.fail_after_chunks == index:
                    raise BackendError(f"injected mid-stream failure for {backend.backend_id}")
                if behavior.chunk_delay:
                    await asyncio.sleep(behavior.chunk_delay)
                yield chunk

        async def close() -> None:
            nonlocal closed
            if not closed:
                closed = True
                self.close_calls[backend.backend_id] = self.close_calls.get(backend.backend_id, 0) + 1

        return StreamHandle(chunks(), close)

    async def complete(self, backend: BackendConfig, path: str, payload: dict[str, object]) -> dict[str, object]:
        del path
        behavior = self._behavior(backend)
        self.open_calls[backend.backend_id] = self.open_calls.get(backend.backend_id, 0) + 1
        if behavior.open_failures > 0:
            behavior.open_failures -= 1
            raise BackendError(f"injected completion failure for {backend.backend_id}")
        return {
            "id": "fake-request",
            "object": "chat.completion",
            "model": payload.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "fake-token"}}],
        }

    async def health(self, backend: BackendConfig) -> bool:
        return self._behavior(backend).healthy

    async def aclose(self) -> None:
        return None
