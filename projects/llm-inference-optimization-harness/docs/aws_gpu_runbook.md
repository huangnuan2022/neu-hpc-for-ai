# AWS GPU Runbook

This runbook is for collecting real CUDA/NCCL benchmark numbers for the LLM inference optimization harness without leaving expensive EC2 resources running.

## Cost Guard Model

AWS is pay-as-you-go, not a strict prepaid card. A `$20` budget does not automatically cap charges by itself. The safe pattern here is:

1. Create a budget alert.
2. Launch a temporary GPU instance with an explicit TTL.
3. Run the benchmark over SSH.
4. Download results locally.
5. Terminate the instance, delete the temporary key pair, and delete the temporary security group.

The helper script defaults to termination cleanup and refuses to launch unless you pass `--confirm-cost YES`.

## Prerequisites

Install and configure the AWS CLI:

```bash
aws configure
aws sts get-caller-identity
```

For a personal AWS account with a scoped IAM user, see:

- `docs/aws_personal_account_setup.md`
- `aws/gpu-benchmark-iam-policy.json`

Your account also needs GPU quota for the chosen instance family. For a first run, use `g5.xlarge` or `g4dn.xlarge`. For a stronger multi-GPU NCCL run, use `g5.12xlarge` or `g4dn.12xlarge`.

## Estimate Cost

```bash
python3 scripts/aws_gpu_benchmark.py estimate --instance-type g5.xlarge --minutes 60
python3 scripts/aws_gpu_benchmark.py estimate --instance-type g5.12xlarge --minutes 60
```

Approximate `us-east-1` On-Demand compute rates used by the script:

| instance | GPUs | approximate rate |
| --- | ---: | ---: |
| `g4dn.xlarge` | 1 T4 | `$0.526/hr` |
| `g5.xlarge` | 1 A10G | `$1.006/hr` |
| `g4dn.12xlarge` | 4 T4 | `$3.912/hr` |
| `g5.12xlarge` | 4 A10G | `$5.672/hr` |

## Create a Budget Alert

```bash
python3 scripts/aws_gpu_benchmark.py create-budget \
  --email YOUR_EMAIL@example.com \
  --limit-usd 20
```

AWS will send confirmation mail for the budget subscription. Confirm it before launching resources.

## Preflight

```bash
python3 scripts/aws_gpu_benchmark.py preflight --region us-east-1
```

This checks your AWS identity and tries to resolve the latest Ubuntu Deep Learning GPU AMI. If AMI auto-detection fails, pick a current Deep Learning GPU AMI in the EC2 console and pass `--ami-id ami-...`.

## Single-GPU Run

Use this first. It is cheap and validates the AMI, SSH, CUDA toolchain, and benchmark path.

```bash
python3 scripts/aws_gpu_benchmark.py launch-run \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --max-runtime-minutes 60 \
  --confirm-cost YES
```

Expected cost for one hour is roughly `$1.02` plus tiny EBS and IPv4 charges.

## Multi-GPU NCCL Run

Run this only after the single-GPU run works.

```bash
python3 scripts/aws_gpu_benchmark.py launch-run \
  --region us-east-1 \
  --instance-type g5.12xlarge \
  --max-runtime-minutes 60 \
  --confirm-cost YES
```

Expected cost for one hour is roughly `$5.70` plus tiny EBS and IPv4 charges.

## Results

Downloaded artifacts are written under:

```text
aws-results/llm-harness-<timestamp>/
```

Useful files:

- `aws-benchmark-results/remote-run.log`
- `aws-benchmark-results/nvidia-smi.txt`
- `aws-benchmark-results/gpu-count.txt`
- `aws-benchmark-results/cpu-harness-results/benchmark_summary.md`
- `aws-benchmark-results/flashattn-1gpu-seq1024.txt`
- `aws-benchmark-results/flashattn-1gpu-seq2048.txt`
- `aws-benchmark-results/flashattn-<N>gpu-seq2048.txt`
- `aws-benchmark-results/flashattn-<N>gpu-seq4096.txt`

After the run, copy the best real GPU metrics into the project README and resume bullets. Do not claim GPU speedup until these files prove it.

## Manual Cleanup Checklist

The script attempts cleanup automatically. If interrupted, check:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=llm-inference-harness" \
  --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Name:Tags[?Key=='Name']|[0].Value}"
```

Then terminate any leftover instance:

```bash
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxxxxxx
```

Also check for leftover key pairs and security groups with names starting with `llm-harness-`.
