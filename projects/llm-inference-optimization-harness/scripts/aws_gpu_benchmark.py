#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/huangnuan2022/neu-hpc-for-ai.git"
PROJECT_PATH = "projects/llm-inference-optimization-harness"
DEFAULT_REGION = "us-east-1"

ON_DEMAND_US_EAST_1 = {
    "g4dn.xlarge": 0.526,
    "g5.xlarge": 1.006,
    "g4dn.12xlarge": 3.912,
    "g5.12xlarge": 5.672,
}


def run(cmd: list[str], *, check: bool = True, input_text: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(shlex.quote(part) for part in cmd))
    return subprocess.run(cmd, check=check, text=True, input=input_text, capture_output=True)


def aws(args: list[str], *, region: str, check: bool = True, input_text: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    return run(["aws", "--region", region, *args], check=check, input_text=input_text)


def require_tool(name: str) -> None:
    if not shutil.which(name):
        raise SystemExit(f"Missing required tool: {name}")


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def json_load_stdout(proc: subprocess.CompletedProcess[str]) -> object:
    text = proc.stdout.strip()
    if not text:
        return None
    return json.loads(text)


def get_account_id(region: str) -> str:
    data = json_load_stdout(aws(["sts", "get-caller-identity"], region=region))
    return str(data["Account"])


def public_ip() -> str:
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=15) as response:
        return response.read().decode("utf-8").strip()


def find_latest_dlami(region: str) -> str:
    filters = [
        "Name=name,Values=Deep Learning*GPU*PyTorch*Ubuntu*",
        "Name=architecture,Values=x86_64",
        "Name=state,Values=available",
    ]
    proc = aws(
        [
            "ec2",
            "describe-images",
            "--owners",
            "amazon",
            "--filters",
            *filters,
            "--query",
            "sort_by(Images,&CreationDate)[-1].ImageId",
            "--output",
            "text",
        ],
        region=region,
    )
    ami_id = proc.stdout.strip()
    if not ami_id or ami_id == "None":
        raise SystemExit(
            "Could not auto-resolve a Deep Learning AMI. Pass --ami-id from the EC2 console."
        )
    return ami_id


def root_device_name(ami_id: str, region: str) -> str:
    proc = aws(
        [
            "ec2",
            "describe-images",
            "--image-ids",
            ami_id,
            "--query",
            "Images[0].RootDeviceName",
            "--output",
            "text",
        ],
        region=region,
    )
    device = proc.stdout.strip()
    if not device or device == "None":
        raise SystemExit(f"Could not resolve root device for AMI {ami_id}")
    return device


def create_key_pair(name: str, key_path: Path, region: str) -> None:
    proc = aws(["ec2", "create-key-pair", "--key-name", name, "--query", "KeyMaterial", "--output", "text"], region=region)
    key_path.write_text(proc.stdout, encoding="utf-8")
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def default_vpc_id(region: str) -> str:
    proc = aws(
        [
            "ec2",
            "describe-vpcs",
            "--filters",
            "Name=isDefault,Values=true",
            "--query",
            "Vpcs[0].VpcId",
            "--output",
            "text",
        ],
        region=region,
    )
    vpc_id = proc.stdout.strip()
    if not vpc_id or vpc_id == "None":
        raise SystemExit("No default VPC found. Pass a security group manually or create a default VPC.")
    return vpc_id


def create_security_group(name: str, region: str) -> str:
    vpc_id = default_vpc_id(region)
    proc = aws(
        [
            "ec2",
            "create-security-group",
            "--group-name",
            name,
            "--description",
            "Temporary SSH access for LLM inference harness benchmark",
            "--vpc-id",
            vpc_id,
            "--query",
            "GroupId",
            "--output",
            "text",
        ],
        region=region,
    )
    group_id = proc.stdout.strip()
    cidr = f"{public_ip()}/32"
    aws(
        [
            "ec2",
            "authorize-security-group-ingress",
            "--group-id",
            group_id,
            "--protocol",
            "tcp",
            "--port",
            "22",
            "--cidr",
            cidr,
        ],
        region=region,
    )
    return group_id


def remote_script(max_runtime_minutes: int) -> str:
    ttl = max(10, int(max_runtime_minutes))
    return f"""#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="$HOME/aws-benchmark-results"
mkdir -p "$RESULTS_DIR"
exec > >(tee "$RESULTS_DIR/remote-run.log") 2>&1

echo "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Scheduling safety shutdown in {ttl} minutes"
sudo shutdown -h +{ttl} "LLM inference benchmark TTL reached" || true

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential cmake git make python3
fi

cd "$HOME"
rm -rf neu-hpc-for-ai
git clone {shlex.quote(REPO_URL)}
cd neu-hpc-for-ai
git rev-parse HEAD | tee "$RESULTS_DIR/git-head.txt"

echo "=== Host ===" | tee "$RESULTS_DIR/host.txt"
uname -a | tee -a "$RESULTS_DIR/host.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "$RESULTS_DIR/nvidia-smi.txt"
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  echo "nvidia-smi not found" | tee "$RESULTS_DIR/nvidia-smi.txt"
  GPU_COUNT=0
fi

echo "$GPU_COUNT" | tee "$RESULTS_DIR/gpu-count.txt"

cd "$HOME/neu-hpc-for-ai/{PROJECT_PATH}"
make clean
make test
make bench
cp -R results "$RESULTS_DIR/cpu-harness-results"

if [ "$GPU_COUNT" -gt 0 ]; then
  cd "$HOME/neu-hpc-for-ai/week_08/dist-flash-attn"
  rm -rf build
  mkdir -p build
  cd build
  cmake .. | tee "$RESULTS_DIR/flashattn-cmake.log"
  make -j"$(nproc)" | tee "$RESULTS_DIR/flashattn-build.log"

  ./dist_flash_attn --seq 1024 --dim 64 --gpus 1 | tee "$RESULTS_DIR/flashattn-1gpu-seq1024.txt"
  ./dist_flash_attn --seq 2048 --dim 64 --gpus 1 | tee "$RESULTS_DIR/flashattn-1gpu-seq2048.txt"

  if [ "$GPU_COUNT" -ge 2 ]; then
    ./dist_flash_attn --seq 2048 --dim 64 --gpus "$GPU_COUNT" | tee "$RESULTS_DIR/flashattn-${{GPU_COUNT}}gpu-seq2048.txt"
    ./dist_flash_attn --seq 4096 --dim 64 --gpus "$GPU_COUNT" | tee "$RESULTS_DIR/flashattn-${{GPU_COUNT}}gpu-seq4096.txt"
  fi
fi

cd "$HOME"
tar -czf "$HOME/aws-benchmark-results.tgz" aws-benchmark-results
echo "Finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""


def ssh_base_args(key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
    ]


def wait_for_ssh(public_dns: str, user: str, key_path: Path, timeout_seconds: int = 900) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        proc = subprocess.run(
            [*ssh_base_args(key_path), f"{user}@{public_dns}", "echo ok"],
            text=True,
            capture_output=True,
        )
        if proc.returncode == 0 and "ok" in proc.stdout:
            return
        time.sleep(10)
    raise RuntimeError("Timed out waiting for SSH")


def scp_from_instance(public_dns: str, user: str, key_path: Path, remote_path: str, local_path: Path) -> None:
    run(
        [
            "scp",
            "-i",
            str(key_path),
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            f"{user}@{public_dns}:{remote_path}",
            str(local_path),
        ]
    )


def launch_run(args: argparse.Namespace) -> None:
    if args.confirm_cost != "YES":
        raise SystemExit("Refusing to launch AWS resources without --confirm-cost YES")
    require_tool("aws")
    require_tool("ssh")
    require_tool("scp")

    region = args.region
    instance_type = args.instance_type
    ami_id = args.ami_id or find_latest_dlami(region)
    run_id = f"llm-harness-{timestamp()}"
    key_name = run_id
    security_group_name = run_id
    key_path = ROOT / "aws-results" / f"{run_id}.pem"
    local_results_dir = ROOT / "aws-results" / run_id
    local_results_dir.mkdir(parents=True, exist_ok=True)

    instance_id: Optional[str] = None
    security_group_id: Optional[str] = None
    created_key = False

    try:
        print(f"Using AMI: {ami_id}")
        create_key_pair(key_name, key_path, region)
        created_key = True
        security_group_id = create_security_group(security_group_name, region)

        user_data = remote_script(args.max_runtime_minutes)
        with tempfile.NamedTemporaryFile("w", delete=False) as fh:
            fh.write(user_data)
            user_data_path = fh.name

        root_device = root_device_name(ami_id, region)
        block_device = json.dumps(
            [
                {
                    "DeviceName": root_device,
                    "Ebs": {
                        "VolumeSize": args.volume_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                }
            ]
        )

        proc = aws(
            [
                "ec2",
                "run-instances",
                "--image-id",
                ami_id,
                "--instance-type",
                instance_type,
                "--key-name",
                key_name,
                "--security-group-ids",
                security_group_id,
                "--user-data",
                f"file://{user_data_path}",
                "--instance-initiated-shutdown-behavior",
                "terminate",
                "--block-device-mappings",
                block_device,
                "--tag-specifications",
                f"ResourceType=instance,Tags=[{{Key=Name,Value={run_id}}},{{Key=Project,Value=llm-inference-harness}},{{Key=AutoTerminate,Value=true}}]",
                "--query",
                "Instances[0].InstanceId",
                "--output",
                "text",
            ],
            region=region,
        )
        instance_id = proc.stdout.strip()
        print(f"Launched instance: {instance_id}")

        aws(["ec2", "wait", "instance-running", "--instance-ids", instance_id], region=region)
        aws(["ec2", "wait", "instance-status-ok", "--instance-ids", instance_id], region=region)

        desc = json_load_stdout(
            aws(
                [
                    "ec2",
                    "describe-instances",
                    "--instance-ids",
                    instance_id,
                    "--query",
                    "Reservations[0].Instances[0].{PublicDnsName:PublicDnsName,PublicIpAddress:PublicIpAddress}",
                ],
                region=region,
            )
        )
        public_dns = desc["PublicDnsName"] or desc["PublicIpAddress"]
        print(f"Public DNS: {public_dns}")

        wait_for_ssh(public_dns, args.ssh_user, key_path)
        print("SSH is ready; waiting for benchmark archive.")

        deadline = time.time() + args.max_runtime_minutes * 60
        archive_path = local_results_dir / "aws-benchmark-results.tgz"
        while time.time() < deadline:
            proc = subprocess.run(
                [*ssh_base_args(key_path), f"{args.ssh_user}@{public_dns}", "test -f ~/aws-benchmark-results.tgz"],
                text=True,
                capture_output=True,
            )
            if proc.returncode == 0:
                scp_from_instance(public_dns, args.ssh_user, key_path, "~/aws-benchmark-results.tgz", archive_path)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(local_results_dir)
                print(f"Downloaded results to {local_results_dir}")
                break
            time.sleep(30)
        else:
            raise RuntimeError("Benchmark archive was not ready before max runtime")

    finally:
        if instance_id and not args.keep_instance:
            aws(["ec2", "terminate-instances", "--instance-ids", instance_id], region=region, check=False)
            print(f"Termination requested for {instance_id}")
            aws(["ec2", "wait", "instance-terminated", "--instance-ids", instance_id], region=region, check=False)
        if security_group_id:
            aws(["ec2", "delete-security-group", "--group-id", security_group_id], region=region, check=False)
            print(f"Deleted temporary security group {security_group_id}")
        if created_key:
            aws(["ec2", "delete-key-pair", "--key-name", key_name], region=region, check=False)
            print(f"Deleted temporary key pair {key_name}")
        if key_path.exists() and not args.keep_key:
            key_path.unlink()
            print(f"Deleted local key {key_path}")


def estimate(args: argparse.Namespace) -> None:
    price = ON_DEMAND_US_EAST_1.get(args.instance_type)
    if price is None:
        print(f"No built-in price for {args.instance_type}; check AWS pricing for {args.region}.")
        return
    hours = args.minutes / 60.0
    compute = price * hours
    ipv4 = 0.005 * hours
    ebs = 0.08 * args.volume_gb * (args.minutes / (60.0 * 24.0 * 30.0))
    print(f"Region assumption: us-east-1 Linux On-Demand")
    print(f"Instance: {args.instance_type} at about ${price:.3f}/hr")
    print(f"Runtime: {args.minutes} minutes")
    print(f"Compute: ${compute:.2f}")
    print(f"Public IPv4: ${ipv4:.4f}")
    print(f"{args.volume_gb}GB gp3 EBS: ${ebs:.4f}")
    print(f"Estimated total: ${compute + ipv4 + ebs:.2f}")


def create_budget(args: argparse.Namespace) -> None:
    require_tool("aws")
    account_id = get_account_id(args.region)
    budget_name = args.name
    budget = {
        "BudgetName": budget_name,
        "BudgetLimit": {"Amount": str(args.limit_usd), "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }
    notifications = [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": threshold,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        }
        for threshold in (50, 80, 95)
    ]
    aws(
        [
            "budgets",
            "create-budget",
            "--account-id",
            account_id,
            "--budget",
            json.dumps(budget),
            "--notifications-with-subscribers",
            json.dumps(notifications),
        ],
        region=args.region,
    )
    print(f"Created budget {budget_name} with alerts to {args.email}. Confirm the email subscription.")


def preflight(args: argparse.Namespace) -> None:
    require_tool("aws")
    version = run(["aws", "--version"], check=False)
    print("AWS CLI:", version.stderr.strip() or version.stdout.strip())
    identity = json_load_stdout(aws(["sts", "get-caller-identity"], region=args.region))
    print("Account:", identity["Account"])
    print("User ARN:", identity["Arn"])
    if args.ami_id:
        print("AMI:", args.ami_id)
    else:
        print("Latest DLAMI:", find_latest_dlami(args.region))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Safely run the LLM inference benchmark on an AWS GPU instance.")
    sub = p.add_subparsers(dest="command", required=True)

    common_region = argparse.ArgumentParser(add_help=False)
    common_region.add_argument("--region", default=DEFAULT_REGION)

    est = sub.add_parser("estimate", help="Estimate short-run cost without touching AWS.", parents=[common_region])
    est.add_argument("--instance-type", default="g5.xlarge")
    est.add_argument("--minutes", type=int, default=60)
    est.add_argument("--volume-gb", type=int, default=80)
    est.set_defaults(func=estimate)

    pre = sub.add_parser("preflight", help="Check AWS identity and resolve the GPU AMI.", parents=[common_region])
    pre.add_argument("--ami-id", default=None)
    pre.set_defaults(func=preflight)

    bud = sub.add_parser("create-budget", help="Create a $50-style monthly AWS Budget email alert.", parents=[common_region])
    bud.add_argument("--email", required=True)
    bud.add_argument("--limit-usd", type=float, default=50.0)
    bud.add_argument("--name", default="llm-harness-cost-guard")
    bud.set_defaults(func=create_budget)

    launch = sub.add_parser("launch-run", help="Launch a temporary GPU instance, run benchmark, download results, terminate.", parents=[common_region])
    launch.add_argument("--instance-type", default="g5.xlarge")
    launch.add_argument("--ami-id", default=None)
    launch.add_argument("--ssh-user", default="ubuntu")
    launch.add_argument("--volume-gb", type=int, default=80)
    launch.add_argument("--max-runtime-minutes", type=int, default=60)
    launch.add_argument("--confirm-cost", default="NO", help="Must be YES to launch resources.")
    launch.add_argument("--keep-instance", action="store_true", help="Debug only. Leaves EC2 instance running/stopped.")
    launch.add_argument("--keep-key", action="store_true", help="Debug only. Keeps local temporary SSH key.")
    launch.set_defaults(func=launch_run)

    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
