from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

import uvicorn

from .config import Settings
from .gateway import create_app


def _wait_for_health(url: str, timeout_seconds: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"fake backend did not become healthy: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a gateway with four deterministic fake backends")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-base-port", type=int, default=8101)
    args = parser.parse_args()

    workers: list[subprocess.Popen[bytes]] = []
    endpoints = [f"http://127.0.0.1:{args.backend_base_port + index}" for index in range(4)]
    try:
        for index, endpoint in enumerate(endpoints):
            env = dict(os.environ)
            env["FAKE_BACKEND_ID"] = f"worker-{index}"
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "llm_serving_system.fake_backend:create_app",
                    "--factory",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.backend_base_port + index),
                    "--log-level",
                    "warning",
                ],
                env=env,
            )
            workers.append(process)
        for endpoint in endpoints:
            _wait_for_health(f"{endpoint}/health")

        os.environ["SERVING_BACKEND_ENDPOINTS"] = ",".join(endpoints)
        app = create_app(Settings.from_env())
        print(f"Gateway: http://{args.host}:{args.port}")
        print("Fake backends: " + ", ".join(endpoints))
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        for worker in workers:
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()


if __name__ == "__main__":
    main()
