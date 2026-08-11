# M2 Scheduler Follow-up Measurement Plan

Status: **Gate 2 GO; Gate 3 scheduler timing NO-GO until native router screening**

## Purpose

The paid run first answers two separate questions:

```text
Is the measurement harness clean?
Is enough of the measured path accessible to request composition to detect 2%?
```

It does not begin with the scheduler matrix.

## Gate 0: local preparation

- unit tests and Ruff pass;
- old K-scaling artifacts carry an interpretation notice;
- validation and metric extraction are outside the isolated CUDA interval;
- count exchange is a separate control-plane interval;
- local actual-route replay is not called an oracle;
- coordinated replay is called `best-found`, never an optimum or bound;
- planner restart best-so-far curves are retained;
- FIFO-selection-control and seeded-random-permutation arms exist;
- selection-control logs a checksum of the eagerly computed discarded plan;
- scheduler and timing modes rotate at repetition granularity.

Build the synthetic scheduler-opportunity profile before renting a GPU:

```bash
python hardware/build_scheduler_screening_profile.py \
  --output results/hardware/contract-followup/cpu-screening
```

The profile computes the FIFO and best-found value of the planner's actual objective:
the sum of rank-maximum receive load over every batch and layer. Gate screening uses
the 25th percentile of the realized objective reduction for each `(K, routing)` cell.
A separate composition-invariant lower bound reports how much of the theoretical
opportunity the planner found. The older single-global-maximum achievability remains a
diagnostic only because it does not match the planner objective. `uniform` is seeded
multinomial routing; the former deterministic balanced generator is retained under the
explicit name `balanced_round_robin`. `request_correlated` uses a seed-dependent
preferred rank and a continuous correlation-strength parameter.

This synthetic profile is a mechanism preflight, not the final workload gate. It may
rule out synthetic cells but cannot authorize paid scheduler timing by itself. Before
Gate 3, collect a native denoising router trajectory from the stock LLaDA2.0-mini
model and run the identical summed-critical-load screen:

```bash
python hardware/collect_llada2_router_trace.py \
  --revision dad945cac317da394b390f82c7b40691d8a881ed \
  --output results/hardware/contract-followup/llada2-router

python hardware/screen_measured_router_trace.py \
  results/hardware/contract-followup/llada2-router \
  --trace-phase native_denoising --all-iterations \
  --output results/hardware/contract-followup/llada2-router-screen
```

The collector reproduces the checkpoint's block-diagonal causal mask. It records
initial width ablations at 1/16/32/64 and a stock 32-step denoising trajectory at block
width 32. Every sparse-layer top-k expert ID, request ID, step index, full compute width,
and remaining masked-position count are retained. It remains supporting measured
routing rather than EP timing or task-quality evidence.

The revision above is the immutable Hugging Face snapshot frozen on 2026-08-07. Do not
replace it with `main` during the campaign.

The measured-route screen assumes a declared expert-to-rank placement and a declared
mapping from requests to source ranks. Its result measures route composition freedom
only. It does not contain EP timing. In particular, its reduction fraction must not be
multiplied by the toy 16-expert/top-2 Gate 2 accessible fraction: the checkpoint uses a
different expert count, top-k, layer count, and kernel/message shape.

Audit the physical opportunity upper bound across batch counts before assigning a
scope to a negative result:

```bash
python hardware/audit_batch_count_sensitivity.py \
  --requests-per-rank 8 --batch-counts 2 4 8 \
  --output results/hardware/contract-followup/batch-count-current

python hardware/audit_batch_count_sensitivity.py \
  --requests-per-rank 16 --batch-counts 2 4 8 16 \
  --output results/hardware/contract-followup/batch-count-extended
```

The audit reports the physical upper bound, FIFO objective, best-found objective,
realized reduction, and objective achievability separately. The multi-batch planner is
a deterministic swap-based coordinate descent and has no optimality guarantee. With
the current 8-request generator, the physical-opportunity all-condition median grows
from 1.41% at two batches to 4.68% at four and 9.32% at eight. For
request-correlated K=64, physical-opportunity seed-p25 grows from 5.31% to 16.73% and
29.62%. Therefore a negative result is explicitly scoped to the measured batch count
and candidate-pool size; best-found reduction is never substituted for the invariant
upper bound.

At eight requests per rank with 16 deterministic restarts, the all-condition medians
for two/four/eight batches are 1.41%/4.68%/9.32% physical opportunity and
1.27%/3.67%/4.60% best-found realized reduction. Achievability falls from 94.87% to
66.41% and 49.18%, so growing physical freedom is not conflated with planner success.
At 16 requests per rank with eight restarts, two/four/eight/sixteen-batch medians are
0.92%/2.33%/6.51%/11.54% physical opportunity and
0.89%/1.91%/4.25%/4.93% best-found reduction; achievability falls from 99.1% to 43.9%.

Keep a separate provisional native-shape roofline:

```bash
python hardware/audit_shape_accessibility.py \
  --output results/hardware/contract-followup/shape-accessibility
```

The metadata explicitly fixes every routed/shared expert MLP to three SwiGLU matmuls
(`gate/up/down`) or `6 * tokens * H * N` FLOPs. The v1 generator already used this
factor for toy and mini, but did not declare it; the provenance correction is recorded
in `docs/accounting_changelog.md`.

LLaDA2-mini now emits the required four combinations:

| path / denominator | K=16 | K=32 | K=64 |
|---|---:|---:|---:|
| assignment / EP-only | 50.18% | 62.05% | 70.38% |
| coalesced / EP-only | 31.31% | 42.53% | 51.81% |
| assignment / full iteration | 61.31% | 62.26% | 63.48% |
| coalesced / full iteration | 41.77% | 42.75% | 44.02% |

The full-iteration scope uses `prefix_length + block_width` model-forward positions
and adds 20 attention layers, the first dense SwiGLU with intermediate 5120, FP32
router projections, and the LM head under the same provisional effective-TFLOPS
assumption. Norm, softmax, activation, packing, and other non-matmul kernels remain
omitted and are declared as such.

**Native scheduler authorization is preregistered against destination-coalesced ×
full-iteration timing.** Coalesced EP-only is a mechanism diagnostic. Assignment-
granular rows are correctness-baseline sensitivity only. At K=64 the primary
provisional accessibility is 44.02%, requiring 9.09% realized reduction for a 4%
screen, rather than 5.68% from the assignment-granular EP-only row. All values remain
sensitivity calculations until native timing replaces them.

## Gate 1: hardware and runtime provenance

Record supported clocks, then request persistence and stable graphics/memory clocks:

```bash
python hardware/gpu_measurement_preflight.py \
  --output results/hardware/contract-followup/preflight.json \
  --graphics-clock <SUPPORTED_MHZ> \
  --memory-clock <SUPPORTED_MHZ> \
  --require-lock
```

Enable NCCL execution-path logging before `torchrun`:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=results/hardware/contract-followup/nccl-%h-%p.log
```

`all_to_all_single` may use grouped NCCL point-to-point operations, so
`NCCL_ALGO`/`NCCL_PROTO` are provenance only and are not treated as proof of the actual
path. The debug log is parsed before making an algorithm/determinism claim. Every run
also records the fused-MoE source hash, candidate configuration file hashes, tensor
shape, and runtime versions. All K/mode combinations warm up before their measured
samples are interpreted.

Separate telemetry records SM/memory clocks, temperature, and power every 0.5 seconds.

Clock-lock failure blocks percent-level scheduler claims, but it does not waste the
rental: timing-harness characterization and native-adapter correctness may continue as
exploratory/correctness evidence. They are labeled unlocked and cannot authorize the
scheduler matrix.

## Gate 2: three-mode timing and accessibility pilot

```bash
torchrun --standalone --nproc-per-node=4 hardware/benchmark_timing_gate_ep4.py \
  --output results/hardware/contract-followup/timing-gate \
  --active-positions 1 16 64 --warmup 3 --repetitions 10 \
  --require-nccl-provenance

python hardware/analyze_timing_gate_ep4.py \
  results/hardware/contract-followup/timing-gate \
  --target-mde-percent 2.0 \
  --screening-profile \
    results/hardware/contract-followup/cpu-screening/scheduler_screening_by_cell.csv \
  --power-safety-multiplier 2.0
```

Modes rotate cyclically inside every repetition:

```text
local_copy
  full local expert compute and shape-matched local tensor copies

nccl_minimal
  identical packing, compute, shape-matched local copies, and unpacking plus
  three real all-to-all calls per layer
  with one element per rank pair

nccl_real
  identical non-collective work, including dead shape-matched local-copy controls,
  plus full hidden dispatch, expert-ID dispatch, and hidden combine
```

The three data-plane collectives are:

```text
1. hidden dispatch
2. expert-ID dispatch
3. hidden combine
```

Split-count all-gather is a fourth, separately timed control-plane collective and is
not included in the three-mode data-plane accounting.

Derived quantities:

```text
launch/synchronization floor = nccl_minimal - local_copy
scheduler-accessible payload = nccl_real - nccl_minimal
accessible fraction          = accessible payload / nccl_real
screened recoverable share   = accessible fraction
                               × p25(realized summed-critical-load reduction)
```

The gate uses a conservative screening condition:

```text
screened recoverable share >= 2 × target MDE
```

Objective achievability is `(FIFO objective - best objective) / (FIFO objective -
composition-invariant lower bound)`. Realized reduction is `(FIFO objective - best
objective) / FIFO objective`. Both come from the existing deterministic route
workload; neither is hard-coded. Screening is performed per `(K, routing)` with the
25th percentile across seeds, so one routing class cannot veto another.

### Three-way result

```text
FAIL
  bootstrap-median lower confidence bounds do not distinguish the NCCL launch floor
  and total NCCL premium from local copy, or
  end-to-end unattributed fraction exceeds 15%
  -> repair harness or stop and return the instance

PASS-UNPOWERED
  harness is clean, but no K has enough accessible recoverable share
  -> skip scheduler matrix and move to native-adapter correctness

PASS-POWERED
  at least one K has enough recoverable share
  -> run only powered scheduler cells
```

Phase attribution has a stricter gate than end-to-end comparison:

```text
maximum unattributed fraction <= 5%
maximum mode/arm unattributed gap < 1 percentage point
```

If only the 15% harness gate passes, end-to-end results may be reported but dispatch,
compute, or combine causality is not claimed.

Repetition screening uses the standard deviation of paired mode differences normalized
by the real-NCCL reference mean. It does not use arm CV or the unstable CV of a
near-zero difference.

For a 2% MDE, two-sided alpha 0.05, and power 0.8, ten repetitions are sufficient only
when the paired-difference SD normalized by the real-NCCL reference mean is at most
approximately 2.26%. Otherwise the analyzer's required-repetition estimate governs.

### Gate 2B: constructed objective-to-time validation

Load-space screening cannot validate itself. During the same rental, run one
preregistered constructed route cell with equal total work. Its FIFO composition is
first measured under three transport controls:

```text
FIFO local-copy control
FIFO minimal-payload NCCL control
FIFO full-payload NCCL baseline
```

This supplies a route-shape-matched accessibility denominator instead of importing
the balanced-route Gate 2 fraction. The general Gate 2 value is retained only as a
cross-check. The full-payload path then evaluates two objective doses:

```text
FIFO constructed plan      = baseline summed critical-load objective
low-dose constructed plan  = exactly 1/12 lower objective (8.333%)
balanced constructed plan  = exactly one-third lower objective
```

Every destination remains non-empty, and total assignments per destination across both
batches are unchanged. Source-specific receive splits are derived from the complete
global plan before timing; no source-homogeneity assumption or host-visible count
extraction occurs inside the CUDA interval. Only per-batch alignment changes. All five
arms rotate by repetition:

```bash
torchrun --standalone --nproc-per-node=4 \
  hardware/benchmark_proxy_validation_ep4.py \
  --output results/hardware/contract-followup/proxy-validation \
  --active-positions 64 --warmup 3 --repetitions 10 \
  --require-nccl-provenance

python hardware/analyze_proxy_validation_ep4.py \
  results/hardware/contract-followup/proxy-validation \
  --timing-gate-analysis \
    results/hardware/contract-followup/timing-gate/timing_gate_analysis
```

The analyzer reports direction and the continuous transmission fraction:

```text
transmission = measured latency reduction
               / (constructed-FIFO accessible fraction × objective reduction)
```

It reports bootstrap intervals for the constructed launch floor, accessible payload,
each dose, and an objective-to-latency slope through the origin. `ALIGNED` alone does
not authorize later screening: the measured transmission estimate and CI become the
conversion coefficient. The two non-zero doses diagnose whether one-point linear
extrapolation is plausible. If the constructed accessibility is not identified, or the
high-dose result is negative or unresolved, a zero-cell load screen is not reported as
absence of timing opportunity; the result instead bounds or rejects the
max-receive-load proxy itself.

## Gate 3A: broad offline composition-freedom matrix

This matrix runs only when both conditions hold:

```text
1. Gate 2 returns PASS-POWERED for the controlled toy EP shape.
2. The native denoising router trace has non-zero, reproducible realized objective
   reduction under the planned contiguous EP=4 placement.
```

Condition 2 is an axis-level workload check, not a numerical timing authorization.
If the native trace has negligible composition opportunity, skip Gate 3A and move to
native EP correctness, active-width, and placement measurements. If it has opportunity,
the exact native shape still requires its own timing-accessibility measurement before a
native scheduler speedup claim.

The controlled matrix prioritizes cell width over online-arm depth:

```text
K:       1, 16, 64
Routing: uniform, mild_skew, strong_skew,
         request_correlated, temporally_unstable
Seeds:   17, 29, 41, 53, 67 for seed-cluster inference
Warmup:  at least 3
Reps:    at least 10, increased only by paired-difference power analysis
Core arms:
  FIFO
  seeded random permutation
  coordinated best-found route replay with zero online planning cost

Dose-response arms in powered cells:
  coordinated_dose_25
  coordinated_dose_50
  coordinated_dose_75
```

The conceptual grid is screened before execution. Each powered `(K, routing)` cell is
invoked separately so an unpowered cross-product cell is never run accidentally and a
cell without a distinct dose ladder fails before measurement.

K=1 is a calibration anchor. It is not expected to be powered for a standalone
scheduler speedup claim.

For every cell retain:

- predicted combined-load reduction;
- FIFO and best-found maximum receive load;
- measured paired latency change;
- full rank-pair split vectors;
- restart 1..N cost and best-so-far curves;
- whether either of the final two restarts improved the best objective.

If the last two restarts still improve, increase deterministic restarts before treating
the best-found plan as evidence of available composition freedom.

Dose plans are valid deterministic request compositions selected near 25%, 50%, and
75% of the FIFO-to-best predicted objective reduction. Candidates come from
source-wise FIFO/best mixtures and seeded random valid plans. Analysis uses the
achieved dose, not the target label. A cell is excluded from dose-slope inference if
request granularity cannot produce at least three distinct positive doses; duplicate
plans are never used to manufacture a ladder.

Predicted reduction is a continuous dose, not a 1% exclusion threshold. The primary
model uses only within-cell variation:

```text
paired latency change
  ~ cell fixed effects
  + achieved predicted-load-reduction dose
```

This prevents K-dependent accessible fraction and routing-dependent imbalance from
identifying the dose coefficient. Accessible fraction and measured imbalance remain
in the raw calibration artifact as substrate context and potential mediators. The
within-cell dose slope and seed-cluster confidence interval are primary. A sign test
across the 15 K/routing cells is secondary; Holm correction still applies to multiple
arm/metric comparisons.

Five seeds are required for a seed-cluster bootstrap confidence interval. A three-seed
pilot may report per-seed slopes only; it cannot be promoted to the final clustered
interval without measuring seeds 53 and 67.

After aggregation, build the preregistered paired calibration artifact with:

```bash
python hardware/analyze_composition_calibration.py \
  results/hardware/contract-followup/scheduler/aggregated \
  --timing-gate-analysis \
    results/hardware/contract-followup/timing-gate/timing_gate_analysis \
  --output results/hardware/contract-followup/composition-calibration
```

## Gate 3B: minimal online-cost confirmation

Online scheduling is not repeated over the broad matrix unless its measured selection
cost is below the Gate 2 recoverable-time bound. Otherwise run only representative
powered K=64 cells to confirm the arithmetic prediction:

```text
FIFO
FIFO + rank-local critical selection discarded
rank-local critical path
local actual-route replay
```

Every arm performs one normal index/gather application in the data path. The selection
control computes the critical plan eagerly, records its checksum, discards it, and then
applies FIFO once. It does not add a second artificial permutation that the actual arm
does not perform.

Online selection, count exchange, and data-plane time are reported separately and
summed for end-to-end latency.

## Statistics

- paired P50 difference by `(seed, K, routing, repetition)`;
- paired-difference bootstrap confidence interval;
- continuous calibration curve with confidence interval;
- sign test across K/routing cells as a secondary robustness statistic;
- Holm correction for multiple comparisons;
- P95/P99 descriptive only without a dedicated high-sample tail run.

## Communication-substrate boundary

NCCL is the controlled baseline, not the performance ceiling. A negative composition
result is scoped to the measured NCCL fixed/payload ratio. Before publication, report a
sensitivity showing how the recoverable fraction changes if an optimized
dispatch/combine substrate reduces fixed communication cost by 2x, 3x, and 5x. A
DeepEP-equivalent microbenchmark is required before claiming substrate-general failure.

The accounting artifact also reports an explicit expert-major `3 -> 2` data-collective
scenario. It reconstructs destination-local expert IDs from expert-level counts and
removes the int32 expert-ID collective. This scopes a negative result to the measured
three-collective implementation without silently changing the primary path.
