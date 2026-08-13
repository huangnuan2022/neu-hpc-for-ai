from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import DEFAULT_MODEL


def _prompt_text(payload: dict[str, object]) -> str:
    if isinstance(payload.get("messages"), list):
        return json.dumps(payload["messages"], sort_keys=True, separators=(",", ":"))
    return str(payload.get("prompt", ""))


def _deterministic_tokens(payload: dict[str, object]) -> list[str]:
    count = min(int(payload.get("max_tokens", 8)), int(os.getenv("FAKE_BACKEND_TOKEN_LIMIT", "16")))
    digest = hashlib.sha256(_prompt_text(payload).encode("utf-8")).hexdigest()[:8]
    return [f"fake-{digest}-{index} " for index in range(count)]


def create_app() -> FastAPI:
    worker_id = os.getenv("FAKE_BACKEND_ID", "fake-worker")
    initial_delay = float(os.getenv("FAKE_BACKEND_INITIAL_DELAY_SECONDS", "0.01"))
    token_delay = float(os.getenv("FAKE_BACKEND_TOKEN_DELAY_SECONDS", "0.005"))
    app = FastAPI(title=f"Deterministic fake vLLM backend ({worker_id})")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "worker_id": worker_id, "fake": True}

    async def respond(request: Request, *, chat: bool) -> Response:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=422, detail="request body must be an object")
        tokens = _deterministic_tokens(payload)
        request_id = f"fake-{hashlib.sha256(_prompt_text(payload).encode()).hexdigest()[:12]}"
        if not payload.get("stream", False):
            if initial_delay:
                await asyncio.sleep(initial_delay)
            text = "".join(tokens)
            choice: dict[str, object]
            if chat:
                choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            else:
                choice = {"index": 0, "text": text, "finish_reason": "stop"}
            return JSONResponse(
                {
                    "id": request_id,
                    "object": "chat.completion" if chat else "text_completion",
                    "created": 0,
                    "model": payload.get("model", DEFAULT_MODEL),
                    "choices": [choice],
                    "usage": {"prompt_tokens": 0, "completion_tokens": len(tokens), "total_tokens": len(tokens)},
                }
            )

        async def events() -> AsyncIterator[bytes]:
            if initial_delay:
                await asyncio.sleep(initial_delay)
            for token in tokens:
                choice = (
                    {"index": 0, "delta": {"content": token}, "finish_reason": None}
                    if chat
                    else {"index": 0, "text": token, "finish_reason": None}
                )
                body = {
                    "id": request_id,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": 0,
                    "model": payload.get("model", DEFAULT_MODEL),
                    "choices": [choice],
                }
                yield f"data: {json.dumps(body, separators=(',', ':'))}\n\n".encode("utf-8")
                if token_delay:
                    await asyncio.sleep(token_delay)
            yield b"data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await respond(request, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await respond(request, chat=False)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic fake vLLM backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
