from __future__ import annotations

from fastapi.testclient import TestClient

from llm_serving_system.fake_backend import create_app


def test_fake_backend_starts_and_streams_deterministically() -> None:
    app = create_app()
    body = {
        "model": "Qwen/Qwen3-8B",
        "messages": [{"role": "user", "content": "deterministic prompt"}],
        "stream": True,
        "max_tokens": 2,
    }

    with TestClient(app) as client:
        health = client.get("/health")
        first = client.post("/v1/chat/completions", json=body)
        second = client.post("/v1/chat/completions", json=body)

    assert health.status_code == 200
    assert first.status_code == 200
    assert first.content == second.content
    assert first.text.count('"content":"fake-') == 2
    assert first.text.endswith("data: [DONE]\n\n")
