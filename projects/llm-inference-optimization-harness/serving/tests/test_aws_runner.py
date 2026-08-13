from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "aws_gpu_benchmark.py"
SPEC = importlib.util.spec_from_file_location("aws_gpu_benchmark", SCRIPT)
assert SPEC and SPEC.loader
aws_runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aws_runner)


def test_generated_remote_script_is_valid_bash_and_archives_on_exit() -> None:
    script = aws_runner.remote_script(
        "all",
        1,
        "vllm/vllm-openai:v0.26.0-cu129-ubuntu2404",
        1.006,
    )
    result = subprocess.run(["bash", "-n"], text=True, input=script, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'RESULTS_DIR="$HOME/aws-benchmark-results"' in script
    assert "trap finalize EXIT" in script
    assert "docker compose version" in script
    assert "llm-serving-benchmark" in script
    assert "run_gpu_sweep.py" in script


def test_user_data_contains_only_ttl_shutdown() -> None:
    user_data = aws_runner.ttl_user_data(60)
    assert "shutdown -h +60" in user_data
    assert "git clone" not in user_data
