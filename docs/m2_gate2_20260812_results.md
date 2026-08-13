# M2 Gate 2/2B H100x4 result

Date: **2026-08-12 KST / 2026-08-11 UTC**

## Decision

```text
Gate 2 timing harness       PASS-UNPOWERED
Gate 2B load-to-time proxy  PROXY_TIME_UNRESOLVED
Gate 3 scheduler matrix     NOT RUN
active Vast.ai instances    0
```

This is an unclamped-clock hardware characterization of the controlled toy shape. It
is not a percent-level performance claim and is not native LLaDA2 evidence.

## Environment and admission

- Four visible NVIDIA H100 80GB HBM3 GPUs, compute capability 9.0.
- Every participating GPU pair reported `NV18`; CUDA peer access passed for all 12
  directed pairs.
- Frozen source-bundle manifest verification passed.
- vLLM 0.27.1, PyTorch 2.13.0+cu130, CUDA 13.0.
- FlashInfer 0.6.16.post3 imported successfully; `FA3_AVAILABLE` was true.
- The actually linked NCCL reported by PyTorch was **2.29.7**, despite the image-build
  metadata candidate listing 2.30.7. Hardware results must use the runtime value.
- NCCL debug logs show direct `P2P/CUMEM` connectivity over the NVLink fabric.
- The exact `E=4, N=8192` H100 fused-MoE tuning file was absent; vLLM used its default
  MoE configuration. Results are substrate/configuration specific.

Clock admission did not pass. Persistence mode was enabled, but the provider denied
graphics-clock locking. The requested memory-lock command was reported unsupported.
Telemetry observed graphics clocks from 345 to 1980 MHz, memory fixed at 2619 MHz,
and maximum temperature 53 C. Consequently, small relative differences are not
eligible for performance claims.

## Correctness smoke

The minimal EP4 smoke executed the frozen 8-layer, 16-expert, top-2, hidden-2048 path
with real dispatch, rank-owned expert compute, and combine. Finite-output,
expert-range, and dispatched/combined assignment-count checks passed. No Gate 3 or
model download was performed.

## Gate 2 result

The corrected three-mode harness retained ten measured repetitions per K after three
warmups. There are 360 measured rank rows:

```text
K = 1, 16, 64
local_copy
nccl_minimal
nccl_real
```

The analyzer reported:

- harness valid;
- positive launch-floor lower confidence bound at every K;
- positive total NCCL-premium lower confidence bound at every K;
- maximum P50 absolute unattributed fraction: 1.171%;
- maximum unattributed mode gap: 0.418 percentage points;
- no powered active-position value and no powered scheduler cell.

Selected critical-rank P50 values:

| K | local copy | NCCL minimal | NCCL real | real payload bytes/layer |
|---:|---:|---:|---:|---:|
| 1 | 6.003 ms | 9.713 ms | 9.510 ms | 262,272 |
| 16 | 5.941 ms | 9.464 ms | 10.067 ms | 4,196,352 |
| 64 | 5.976 ms | 9.335 ms | 9.403 ms | 16,785,408 |

At K=64, the estimated launch floor was 3.330 ms, while the full-payload-minus-minimal
median was only 0.066 ms and its bootstrap CI crossed zero (`-2.009` to `1.648` ms).
The controlled shape is therefore fixed-cost dominated at the resolution achieved by
this run. It does not support a scheduler matrix or a K-scaling speedup claim.

## Gate 2B result

Gate 2B retained five arms and source-specific split contracts:

```text
FIFO local copy
FIFO minimal payload
FIFO full payload
8.333% objective-reduction dose
33.333% objective-reduction dose
```

Its constructed-FIFO accessible-payload CI also crossed zero (`-0.921` to `2.156`
ms), so transmission could not be estimated. The 33.333% dose produced a median
GPU-path reduction of `-0.044%` with a wide bootstrap interval (`-22.78%` to `8.63%`).
The objective-to-latency slope CI also crossed zero.

The correct status is `PROXY_TIME_UNRESOLVED`, not `DISCONFIRMED`. Therefore:

- a load-space null result cannot be promoted to “no timing opportunity”;
- no claim may be made that planner-objective reduction helps or does not help;
- the next decision must use the native shape and measured native routes.

## Cost and teardown

The instance ran for at most approximately 15.62 minutes from recorded start until
post-destroy confirmation. At $26.7478/hour, the conservative compute-cost estimate is
approximately **$6.96**, below the $13.36 ceiling. Forty-two raw artifact files were
copied locally before teardown, and Vast.ai reported zero active instances afterward.

## Next step

Do not repeat Gate 2 with more toy repetitions and do not run Gate 3. The prescribed
stock LLaDA2.0-mini route collection was completed on 2026-08-13; see the
[M2.1 result](m2_1_20260813_results.md). Its best-found request-composition headroom is
below 0.5% in every measured cell, so paid native-shape scheduler timing is not
authorized by the current evidence. LLaDA-MoE-7B remains only a later true-EP adapter
plumbing smoke target.

Raw and analyzed artifacts are under:

`results/hardware/h100_ep4_20260812_gate2/contract-followup/`
