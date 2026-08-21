from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


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
    assert "source /opt/pytorch/bin/activate" in script
    assert "apt-get install -y cuda-toolkit libnccl2 libnccl-dev" in script
    assert '-DCMAKE_CUDA_COMPILER="$NVCC_PATH"' in script
    assert "-DCMAKE_CUDA_ARCHITECTURES=86" in script
    assert "docker compose version" in script
    assert "llm-serving-benchmark" in script
    assert "run_gpu_sweep.py" in script


def test_user_data_contains_only_ttl_shutdown() -> None:
    user_data = aws_runner.ttl_user_data(60)
    assert "shutdown -h +60" in user_data
    assert "git clone" not in user_data


def test_cost_estimate_uses_requested_volume_size() -> None:
    compute, public_ipv4, gp3 = aws_runner.estimated_cost_components(5.672, 60, 120)
    assert compute == pytest.approx(5.672)
    assert public_ipv4 == pytest.approx(0.005)
    assert gp3 == pytest.approx(0.013333333333333334)


def test_run_surfaces_failure_output(capsys: pytest.CaptureFixture[str]) -> None:
    result = aws_runner.run(
        ["bash", "-c", "printf visible-out; printf visible-error >&2; exit 7"],
        check=False,
    )
    captured = capsys.readouterr()
    assert result.returncode == 7
    assert "visible-out" in captured.err
    assert "visible-error" in captured.err
