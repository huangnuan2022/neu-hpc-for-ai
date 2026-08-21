# AWS Four-A10G Measurement Notes

## Evidence Scope

- Source commit: `e056e29e326868f7a88405f6ae13b3a02f2d3bf7` with a clean worktree.
- Instance: `g5.12xlarge` in `us-east-1c`, with four NVIDIA A10Gs.
- Instance ID: `i-0108995b4a93240a1`; launch time `2026-08-21T09:15:23Z`; termination requested at approximately `2026-08-21T09:37:01Z`.
- AMI: `ami-0fe19eeadb42a3627`; NVIDIA driver `595.91.07`; CUDA toolkit `13.3`; PyTorch `2.10.0+cu130`; NCCL `2.31.2`.
- Model: `Qwen/Qwen3-8B`, BF16.
- Serving image: `vllm/vllm-openai:v0.26.0-cu129-ubuntu2404`, digest `sha256:f21f5e1987142d4a7c77a4fe41726ab2910153c5253df61e883771258db59440`.
- Measured serving requests: 285, all successful, all with exactly 512 input and 128 output tokens.
- CUDA method: five warm-ups followed by 20 measured iterations for each 1/2/4-GPU, sequence-length, and overlap configuration.

The serving path uses four independent vLLM workers behind the FastAPI gateway. The custom CUDA/NCCL attention executable is a separate microbenchmark and is not integrated into vLLM.

## Serving Interpretation

At concurrency 32, the four-worker deployment produced `786.408` output tokens/s, `243 ms` p95 TTFT, `39.2 ms` p95 TPOT, and `5.195 s` p95 E2E with no errors or timeouts. Its estimated cost was `$2.003` per million output tokens based on the full `$5.672/hour` instance price and measured workload time; model download, setup, and idle time are excluded.

Compared with the committed one-A10G artifact using the same model, image digest, token lengths, and run counts, throughput increased `6.29%`, p95 TTFT decreased `30.99%`, and p95 E2E decreased `6.18%`. Four replicas do not imply four-times serving throughput: each replica has an independent batcher and prefix cache, while this workload intentionally reuses prefixes. The gateway preserved prefix affinity for 75% of concurrency-32 requests and overrode it for the remaining 25% to control queue imbalance.

## CUDA/NCCL Interpretation

At sequence length 4,096 and dimension 64, the in-run one-GPU median was `18.348 ms`, the two-GPU overlap median was `11.092 ms`, and the four-GPU overlap median was `7.380 ms`. The four-GPU result is a measured `2.49x` speedup over the in-run one-GPU baseline. Separate-stream, double-buffered K/V rotation reduced four-GPU median latency by `6.49%` compared with the serialized control (`7.892 ms`).

The sequence-128 correctness sweep compared complete outputs with PyTorch SDPA and reported at most `2.235e-8` maximum absolute error across 1/2/4 GPUs. The sequence-4,096 performance sweep uses the custom single-GPU implementation as its reference and reported `4.191e-8` maximum absolute error for four GPUs.

At sequence length 4,096, the `98.8159%` minimal-state reduction is formula-based. The `98.0347%` explicit-workspace reduction uses the exact sum of requested CUDA allocations. Observed allocation deltas are retained separately in the JSON and are not presented as the same metric.

The Nsight kernel summary attributes `84.6%` of aggregate GPU kernel time to the attention step and `15.4%` to NCCL send/receive. The trace includes setup and teardown API calls; benchmark steady-state latency uses CUDA events after persistent buffers, streams, events, and communicators are initialized.

## Cost And Cleanup

The instance ran for about 21 minutes 38 seconds before termination was requested. Using the runner's reviewed `$5.672/hour` compute price, `$0.005/hour` public IPv4 estimate, and a prorated 120 GB gp3 volume, the resource cost is estimated at `$2.05` before credits. This is an estimate, not a posted billing charge.

After artifact download, EC2 reported the instance as `terminated`; no attached volume remained. Temporary security group `sg-01f56d0935b2cec98`, key pair `llm-harness-20260821T091510Z`, and the local private key were deleted. A preceding failed launch created no instance and also removed its temporary key and security group.
