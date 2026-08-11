# External Design Review Disposition

Date: **2026-08-09**

## Decision

The full scheduler matrix is **NO-GO** until a short three-mode timing gate passes and
a native LLaDA2 denoising router trace shows reproducible composition opportunity. The
timing gate itself is **GO** because its harness/K-scaling result does not depend on the
route workload. No rental is active.
This does not invalidate the functional EP=4 bring-up or the end-to-end rejection of
the current rank-local online scheduler. It invalidates the prior performance
interpretation of the near-flat K sweep.

## Accepted blockers

### Timing identifiability and accessible time

The v1 `31 ms` value is the critical-rank time for two batches through eight MoE
layers, not one position or one layer. The interval also includes per-layer count
exchange, host-visible tensor extraction, synchronization, correctness checks, and
metric collection.

The next run contains three repetition-rotated modes:

```text
local_copy   = full local compute and shape-matched local copies
nccl_minimal = same compute plus minimal-payload real collectives
nccl_real    = full-payload real dispatch, expert-ID dispatch, and combine
```

This separates the collective launch/synchronization floor from the payload- and
imbalance-accessible term. All modes now execute identical shape-matched local-copy
control work, including `nccl_real`, so the payload difference does not subtract HBM
copy time. Packing, local-copy memory, and unpacking are recorded separately. We do
not force a larger layer or batch shape merely to
make variable cost dominant, because that could leave the decode regime being
studied.

The gate has three outcomes. `FAIL` repairs the harness, `PASS-UNPOWERED` skips the
scheduler matrix and moves to native-adapter correctness, and `PASS-POWERED` permits
only K cells whose screened recoverable share exceeds twice the target MDE.

### Clock and runtime control

The next run attempts persistence mode and fixed graphics/memory clocks before
measurement. Run-level SM clock, memory clock, temperature, and power telemetry is
retained. Clock-lock failure blocks percent-level scheduler claims but does not block
timing characterization or correctness work, which remain explicitly exploratory.

NCCL algorithm/protocol and fused-MoE source/configuration provenance are recorded.
Warmup precedes interpretation of every relevant shape.

### Coordinated-plan diagnostics

For every replay cell, retain FIFO and best-found objective values, combined maximum
receive load, request reassignment fraction, restart costs, and the restart
best-so-far curve. A replay plan is a diagnostic best-found plan, not an oracle or
upper bound. Restarts increase deterministically while either of the final two
restarts improves the best objective, up to the preregistered cap.

Predicted load reduction is a continuous within-cell dose against measured paired
latency change; it is not reduced to a binary vacuity threshold. The gate multiplies
measured accessible fraction by the 25th percentile of the realized reduction in the
planner's actual objective for each `(K, routing)` cell. That objective is the sum of
the maximum destination-rank load over every `(batch, layer)`. The former
single-global-maximum achievability metric is diagnostic only and no longer controls
the gate.

Synthetic screening can reject synthetic cells but cannot authorize paid scheduler
timing on its own. A stock LLaDA2 native denoising route trajectory must first be
screened step-by-step under the planned expert placement. The route-only result
establishes workload opportunity, not EP timing; the native shape still needs its own
measured accessibility before a native scheduler claim.

## Accepted controls

- FIFO composition with rank-local critical selection performed and discarded;
- seeded random request permutation;
- offline coordinated replay with plan-generation time reported separately;
- scheduler arms and timing modes interleaved at repetition granularity;
- full send/receive split vectors and rank-arrival/collective timing;
- a checksum proving the discarded plan was eagerly computed;
- NCCL and fused-MoE provenance captured with each artifact.

## Statistical gate

The old three-repetition data has a median within-cell GPU-path CV of 3.34%, with a
90th percentile of 5.36%. Those estimates include v1 timing contamination and do not
justify choosing 20 repetitions. Repetition screening uses the standard deviation of
paired differences normalized by the real-NCCL reference mean, not arm-level CV or
the unstable CV of a near-zero difference.

P50 with paired differences is primary. P95/P99 from small samples remain descriptive
and cannot support a tail claim.

## Clarifications and rejected overstatements

- More bytes accompanied by a slightly lower measured time is possible when fixed
  collective, launch, synchronization, and host-control costs dominate. It remains a
  diagnostic signal that the bandwidth term was not identified in v1.
- The 42,336 aggregated layer records do not double-count dispatch and combine. Each
  run contains two request batches times eight layers: 16 records per run.
- High K activating all 16 experts does not make routing conditions equal. Rank-load
  imbalance and split vectors determine whether skew remains active.
- The v2 data plane has three collectives per layer: hidden dispatch, expert-ID
  dispatch, and hidden combine. Split-count all-gather is a fourth, separately timed
  control-plane collective. Treating the data plane as two collectives would omit a
  real int32 transfer.
- `FIFO + selection discarded` computes the critical plan eagerly, records its
  checksum, discards it, and applies FIFO once. Adding a second artificial permutation
  would charge work the actual arm does not perform; this Python/PyTorch path is eager.

## Second-review matrix allocation

The broad experiment is an offline composition-freedom diagnostic, not a repeat of
every online arm:

```text
K:       1, 16, 64
Routing: uniform, mild_skew, strong_skew,
         request_correlated, temporally_unstable
Arms:    FIFO, random permutation, coordinated dose 25/50/75,
         coordinated best-found replay
```

K=1 remains a calibration anchor. At least ten paired repetitions are used only after
the timing gate reports a powered regime. Online arms are restricted to a minimal
K=64 confirmation unless measured selection cost is below the accessible-time bound.
A cell-wise sign test is secondary because K/routing cells are not guaranteed
independent; cell-fixed-effect dose response with a five-seed cluster bootstrap is
primary.

The matrix is not executed merely because a synthetic cell passes. It additionally
requires non-zero, reproducible composition opportunity in the native route screen.
The toy timing accessible fraction is never multiplied by the native LLaDA2 reduction
fraction because the expert count, top-k, layers, and messages differ.

## Evidence status after review

```text
CONFIRMED
- real four-rank route -> dispatch -> local expert -> combine path executed
- count, route-range, finite-value, and replay checks passed in v1
- current rank-local online scheduler loses end-to-end in 0/49 improving cells

UNCALIBRATED / NOT CLAIMABLE
- K=1..64 useful-throughput scaling
- magnitude of GPU-only scheduler deltas
- communication-fraction decrease as an amortization benefit

PENDING
- corrected isolated data-path K scaling
- constructed objective-to-measured-time proxy validation
- stock LLaDA2 native denoising router opportunity screen
- coordinated composition freedom and measured calibration slope
- native LLaDA correctness and useful-progress measurements
```

## Communication-substrate boundary

NCCL is the controlled mechanism baseline, not a performance ceiling. A negative
composition result is scoped to the measured NCCL fixed/payload ratio. Before a
substrate-general negative claim, report sensitivity for 2x, 3x, and 5x reductions in
fixed communication cost and run a DeepEP-equivalent microbenchmark.

## 2026-08-09 accounting and proxy amendments

The controlled toy and native-mini roofline audit now declares three SwiGLU matmuls
(`gate`, `up`, `down`) and two FLOPs per multiply-accumulate. The v1 generator already
used this factor-six formula for both profiles; the missing matmul-count metadata was a
provenance defect, not a toy-only formula change. The correction is recorded in
`docs/accounting_changelog.md`.

Native sensitivity is reported as four separate rows:

```text
assignment-granular vs destination-coalesced communication
                    x
EP-only vs full-iteration denominator
```

Native scheduler authorization is preregistered against
`destination-coalesced x full-iteration`. Assignment-granular rows are correctness
sensitivity only, and coalesced EP-only is a mechanism diagnostic. The full-iteration
roofline includes attention matmuls, the first dense layer, router projection, sparse
and shared experts, and LM head. It remains a sensitivity estimate, not a measurement.

Gate 2B now contains two non-zero objective doses, exactly 8.333% and 33.333%, plus
FIFO. It reports not only direction but the continuous transmission fraction

```text
measured latency reduction
---------------------------------------------
measured accessible fraction x objective dose
```

with bootstrap intervals and a two-dose through-origin slope. Directional alignment
alone cannot authorize a later timing screen.

The stock LLaDA2 trajectory does not have confidence-dependent model compute width.
The block remains width 32 and the implementation recomputes clean prefix plus current
block with `use_cache=False`. Trace artifacts therefore separate controlled initial
width, fixed native block width, model-forward positions, remaining masked positions,
and finalized progress.
