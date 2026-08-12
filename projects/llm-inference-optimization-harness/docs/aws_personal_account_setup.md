# Personal AWS Account Setup

Use this when AWS Academy / voclabs cannot provide GPU instances. Do not paste access keys into chat.

## 1. Create an IAM user

1. Open AWS Console with your personal AWS account.
2. Go to IAM -> Users -> Create user.
3. User name: `codex-gpu-benchmark`.
4. Choose direct policy attachment or create an inline policy.
5. Paste the policy from:

```text
projects/llm-inference-optimization-harness/aws/gpu-benchmark-iam-policy.json
```

This policy allows the benchmark runner to create and terminate temporary EC2 instances, temporary SSH keys, temporary security groups, read quotas, check identity, and create a budget alert. It does not grant broad administrator access.

## 2. Create an access key

1. Open the user `codex-gpu-benchmark`.
2. Security credentials -> Create access key.
3. Use case: Command Line Interface.
4. Save the Access Key ID and Secret Access Key somewhere secure.

## 3. Configure AWS CLI locally

Run this in your own Terminal:

```bash
aws configure --profile personal-gpu
```

Enter:

```text
AWS Access Key ID: <your key>
AWS Secret Access Key: <your secret>
Default region name: us-east-1
Default output format: json
```

Then verify:

```bash
AWS_PROFILE=personal-gpu aws sts get-caller-identity
```

If identity works, use the profile with this project:

```bash
export AWS_PROFILE=personal-gpu
cd /Users/huangnuan/Documents/neu-hpc-for-ai/projects/llm-inference-optimization-harness
python3 scripts/aws_gpu_benchmark.py preflight --region us-east-1
```

## 4. Cost guard and benchmark

Create a budget alert:

```bash
python3 scripts/aws_gpu_benchmark.py create-budget \
  --email YOUR_EMAIL@example.com \
  --limit-usd 20
```

Run cheap single-GPU first:

```bash
python3 scripts/aws_gpu_benchmark.py launch-run \
  --region us-east-1 \
  --instance-type g5.xlarge \
  --max-runtime-minutes 60 \
  --confirm-cost YES
```

Only after single-GPU succeeds, run 4-GPU:

```bash
python3 scripts/aws_gpu_benchmark.py launch-run \
  --region us-east-1 \
  --instance-type g5.12xlarge \
  --max-runtime-minutes 60 \
  --confirm-cost YES
```

## 5. Cleanup

The script terminates instances automatically. If anything is interrupted:

```bash
AWS_PROFILE=personal-gpu aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=llm-inference-harness" \
  --query "Reservations[].Instances[].{Id:InstanceId,State:State.Name,Type:InstanceType,Name:Tags[?Key=='Name']|[0].Value}"
```

Terminate any leftover instance:

```bash
AWS_PROFILE=personal-gpu aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxxxxxx
```
