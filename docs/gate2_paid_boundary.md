# Gate 2 paid boundary

Date: **2026-08-12**

## Current state

```text
Gate 0 CPU/static validation     PASS
Gate 2 timing characterization  READY
Gate 2B proxy validation        READY
Gate 3 scheduler matrix         NO-GO
active Vast.ai instances        0
```

The next action that changes external state is instance creation. Nothing in this
document authorizes it.

## Frozen candidate

```text
Offer ID:        36346698 (point-in-time; must be rechecked)
Visible GPUs:    4x H100 SXM requested
Reported NVLink: 478.116
Driver/CUDA max: 580.126.20 / 13.0
Price:           $26.7213889/hour total
Hard stop:       30 minutes proposed
Compute ceiling: $13.36069445 proposed

Image: vllm/vllm-openai:v0.27.1-x86_64
Digest: sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2
CUDA: 13.0.3
NCCL: 2.30.7
FlashInfer: 0.6.16.post3
```

The offer may expose four GPUs from an eight-GPU host. Only the four visible GPUs are
part of the run. Full pairwise NVLink is a runtime admission check, not something
inferred from the marketplace bandwidth field.

## Mandatory order after purchase

1. Start the SSH/direct instance with the immutable image; do not start `vllm serve`.
2. Verify the image digest and frozen source bundle manifest.
3. Save environment, GPU, topology, NCCL, FlashInfer, and FA3 metadata.
4. Lock clocks and start telemetry.
5. Run four-rank correctness smoke.
6. Run Gate 2.
7. Run Gate 2B only if Gate 2 has not failed.
8. Stop telemetry, copy every raw artifact, and destroy the instance.

Do not download LLaDA, run Gate 3, or add an exploratory benchmark during this rental.

## Immediate teardown conditions

- not exactly four visible H100 SXM GPUs;
- missing pairwise NVLink between any participating GPU;
- source bundle manifest mismatch;
- NCCL/token-conservation failure;
- Gate 2 timing status `FAIL`;
- setup consumes ten minutes without reaching correctness smoke;
- wall clock reaches 30 minutes.

Clock lock failure disables percent-level claims but does not prevent correctness and
harness characterization. FA3 failure blocks later native-model reuse of the image but
does not invalidate Gate 2, whose measured path contains no attention kernel.
