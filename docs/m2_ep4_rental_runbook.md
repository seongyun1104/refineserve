# M2 H100x4 Gate 2 rental runbook

Status: **READY FOR OFFER FREEZE; Gate 2 ONLY**

This runbook is subordinate to `hardware_execution_contract.md`. It does not run an
ordinary AR MoE model, framework benchmark, or scheduler matrix.

## Frozen experiment

```text
hardware: one host with exactly 4x H100 SXM 80 GB
fabric: full intra-node NVLink connectivity
EP=4, TP=1, PP=1
layers=8, experts=16, top-k=2
hidden=2048, intermediate=8192, BF16
K=1,16,64
modes=local_copy,nccl_minimal,nccl_real
warmup=3, measured repetitions=10
```

The image digest, driver, CUDA, PyTorch, NCCL, offer ID, hourly price, and maximum
spend are printed and frozen immediately before purchase. Do not reuse the old Qwen
AR/vLLM campaign configuration in this run.

The current unpaid candidate is recorded in
`results/preflight_20260812/paid-run-candidate.json`. It is a point-in-time record,
not permission to create an instance. Re-query the offer immediately before purchase.

The frozen image candidate is:

```text
vllm/vllm-openai:v0.27.1-x86_64
amd64 digest: sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2
CUDA 13.0.3, NCCL 2.30.7, FlashInfer 0.6.16.post3
```

The vLLM v0.27.1 source builds separate FA2 and FA3 extensions and prefers FA3 on
SM90. Runtime FA3 availability and backend selection must still be captured on the
H100 host. FA3 is not used by the Gate 2/2B fused-expert and NCCL measurement path;
this check preserves the environment contract for later native-model work.

## Admission and teardown gates

1. The offer is one four-GPU machine, not four unrelated instances.
2. Every device is H100 SXM with compute capability 9.0 and at least 75 GiB.
3. `nvidia-smi topo -m` reports NVLink on every GPU pair.
4. Four-rank NCCL all-to-all smoke and token conservation pass.
5. Persistence and requested graphics/memory clocks lock successfully for percent-level
   claims. If locking fails, continue harness characterization only and label it.
6. The NCCL debug log and fused-MoE implementation hashes are recorded. The log, not
   `NCCL_ALGO`/`NCCL_PROTO`, is authoritative for the all-to-all execution path.
7. Raw artifacts are copied off-host before teardown.
8. The image digest, FlashInfer import/version, FA3 extension availability, and selected
   attention backend are recorded. Failure of FA3 does not alter Gate 2 data-path
   semantics, but it blocks reuse of this environment for the later native-model run.

A topology or correctness failure triggers immediate copy-and-teardown. A timing-gate
`FAIL` also stops performance work; do not debug the scheduler on rented time.

## Execution order

```text
P0  environment/topology metadata and clock preflight
P1  four-rank NCCL correctness smoke
P2  three-mode timing Gate 2
P2b constructed planner-objective -> time proxy validation
P3  copy raw timing, NCCL logs, telemetry, and profiler evidence
P4  run analyzers off the critical GPU path
P5  mandatory teardown
```

Gate 3 scheduler timing is forbidden in this rental unless it was separately
preregistered after native router screening. LLaDA2 checkpoint download is also not part
of this four-GPU Gate 2 run; use the cheaper single-GPU trace runbook.

## Commands

Freeze the exact local code/document bundle before instance creation:

```bash
python hardware/build_gate2_bundle_manifest.py \
  --output results/hardware/contract-followup/gate2-bundle-manifest.json
```

Copy this manifest with the source tree and verify it on the rental host before
measurement. Any mismatch stops the run:

```bash
python hardware/build_gate2_bundle_manifest.py \
  --verify results/hardware/contract-followup/gate2-bundle-manifest.json
```

Preflight:

```bash
python hardware/preflight_ep4.py \
  --output results/hardware/contract-followup/preflight.json \
  --expected-gpus 4 --require-h100 --require-nvlink

python hardware/gpu_measurement_preflight.py \
  --output results/hardware/contract-followup/clock-preflight.json \
  --graphics-clock <SUPPORTED_MHZ> \
  --memory-clock <SUPPORTED_MHZ> \
  --require-lock
```

Enable NCCL path logging:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=results/hardware/contract-followup/nccl-%h-%p.log
```

Start telemetry before either measured gate and stop it immediately afterward:

```bash
TELEMETRY_STOP=results/hardware/contract-followup/telemetry.stop
python hardware/gpu_telemetry.py \
  --output results/hardware/contract-followup/gpu-telemetry.csv \
  --interval-seconds 0.5 --duration-seconds 3600 \
  --stop-file "$TELEMETRY_STOP" &
TELEMETRY_PID=$!
```

The stop file must not exist before launch. After Gate 2B, run `touch
"$TELEMETRY_STOP"` and `wait "$TELEMETRY_PID"` before copying artifacts.

`torch.distributed.all_to_all_single` may be lowered to grouped point-to-point
send/receive operations. `NCCL_ALGO` and `NCCL_PROTO` are recorded if present but are
not treated as proof that Ring/LL controls this path. Parse the debug log before making
a determinism or algorithm claim.

Create the CPU screening artifact before launch or copy the already validated artifact:

```bash
python hardware/build_scheduler_screening_profile.py \
  --output results/hardware/contract-followup/cpu-screening
```

Run Gate 2:

```bash
torchrun --standalone --nproc-per-node=4 \
  hardware/benchmark_timing_gate_ep4.py \
  --output results/hardware/contract-followup/timing-gate \
  --active-positions 1 16 64 \
  --warmup 3 --repetitions 10 \
  --require-nccl-provenance
```

Run the preregistered constructed proxy cell before teardown. It includes FIFO
local-copy/minimal/full-payload controls plus exact 8.333% and 33.333%
objective-reduction doses. Total work, expert-ID element count, and global destination
assignment totals remain fixed. Source-specific send/receive splits are precomputed
from the full four-rank plan:

```bash
torchrun --standalone --nproc-per-node=4 \
  hardware/benchmark_proxy_validation_ep4.py \
  --output results/hardware/contract-followup/proxy-validation \
  --active-positions 64 --warmup 3 --repetitions 10 \
  --require-nccl-provenance

Analyze after copying raw artifacts:

```bash
python hardware/analyze_timing_gate_ep4.py \
  results/hardware/contract-followup/timing-gate \
  --target-mde-percent 2.0 \
  --screening-profile \
    results/hardware/contract-followup/cpu-screening/scheduler_screening_by_cell.csv

python hardware/analyze_proxy_validation_ep4.py \
  results/hardware/contract-followup/proxy-validation \
  --timing-gate-analysis \
    results/hardware/contract-followup/timing-gate/timing_gate_analysis
```

## Decision

```text
FAIL
  copy artifacts, stop performance work, teardown

PASS-UNPOWERED
  preserve the shape-specific negative boundary, skip scheduler matrix, teardown

PASS-POWERED
  record the powered controlled cells, but do not run Gate 3 until the separate native
  router-opportunity prerequisite is satisfied
```

The Gate 2 result is valid for the controlled 16-expert/top-2 shape only. It cannot be
multiplied by LLaDA2's 256-expert/top-8 route reduction as if the shapes matched.

The constructed proxy result controls the negative interpretation:

```text
PROXY_TIME_ALIGNED
  use the constructed-FIFO accessibility and measured transmission CI, not direction
  alone or the balanced-route Gate 2 fraction, to convert a load reduction into a
  shape-scoped timing screen

PROXY_TIME_DISCONFIRMED or PROXY_TIME_UNRESOLVED
  load-space 0-cell results cannot be promoted to a timing-opportunity claim
```

## Cost control

Before instance creation, print together:

```text
offer ID
host GPU count and topology claim
image name and immutable digest
hourly price
hard-stop duration
maximum authorized spend
```

Do not begin a model download or an unregistered matrix to use remaining time. Teardown
is mandatory even if analysis is deferred.

The current proposal is a 30-minute hard stop. At the observed total price of
`$26.7213889/hour`, the compute ceiling is `$13.36069445`, excluding any separately
billed storage or transfer. This is a proposal only; the price and explicit purchase
authorization must be confirmed at the paid boundary.
